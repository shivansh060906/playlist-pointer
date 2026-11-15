"""RAGEngine: owns the models and per-strategy retrievers; the API and scripts both go through it."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

from .config import STRATEGIES, Settings, get_settings
from .index.store import chunks_path, index_exists, load_chunks
from .library import Library
from .llm import LLMClient, build_messages, cited_indices
from .retrieval.embedder import SentenceTransformerEmbedder
from .retrieval.glossary import Glossary
from .retrieval.hybrid import Hit, Retriever, SearchResult
from .retrieval.reranker import CrossEncoderReranker


@dataclass
class AskResult:
    answer: str
    hits: list[Hit]
    cited: list[int]  # 1-based indices into hits
    timings_ms: dict[str, float] = field(default_factory=dict)
    model: str = ""
    rerank_applied: bool = False
    out_of_scope: bool = False
    top_dense: float | None = None


OUT_OF_SCOPE_ANSWER = (
    "That doesn't look related to this playlist, so I won't guess. "
    "Try asking about something the videos actually cover."
)
NOT_COVERED_ANSWER = "This playlist doesn't seem to cover that."


class RAGEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder=None,
        reranker=None,
        llm: LLMClient | None = None,
        glossary: Glossary | None = None,
    ):
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self.glossary = glossary if glossary is not None else Glossary.load()
        self._retrievers: dict[tuple[str, str], Retriever] = {}
        self._lock = threading.RLock()
        self.library = Library(self.settings, lambda: self.embedder, on_change=self._forget)

    def _forget(self, playlist: str) -> None:
        """Drop cached retrievers for a playlist (after it was (re)built or deleted)."""
        with self._lock:
            for key in [k for k in self._retrievers if k[0] == playlist]:
                del self._retrievers[key]

    # -- lazy components --------------------------------------------------------------------
    @property
    def embedder(self):
        with self._lock:
            if self._embedder is None:
                self._embedder = SentenceTransformerEmbedder(self.settings.embed_model)
            return self._embedder

    @property
    def reranker(self):
        with self._lock:
            if self._reranker is None:
                self._reranker = CrossEncoderReranker(self.settings.rerank_model)
            return self._reranker

    @property
    def llm(self) -> LLMClient:
        with self._lock:
            if self._llm is None:
                self._llm = LLMClient.from_settings(self.settings)  # raises LLMNotConfigured
            return self._llm

    def retriever(self, strategy: str | None = None, playlist: str | None = None) -> Retriever:
        strategy = strategy or self.settings.default_strategy
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        playlist = playlist or "default"
        settings = self.library.settings_for(playlist)  # ValueError for unknown playlists
        with self._lock:
            key = (playlist, strategy)
            if key not in self._retrievers:
                self._retrievers[key] = Retriever.load(
                    settings, strategy, self.embedder, self.reranker, self.glossary
                )
            return self._retrievers[key]

    def warmup(self) -> None:
        """Load models + default index so the first real request is fast. Never raises."""
        try:
            retriever = self.retriever()
            retriever.search("warmup", k=1, rerank=False, expand=False)  # don't load the cross-encoder
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup] skipped: {exc}")

    # -- info -------------------------------------------------------------------------------
    def available_strategies(self, playlist: str | None = None) -> list[str]:
        st = self.library.settings_for(playlist)
        return [s for s in STRATEGIES if index_exists(st, s)]

    def llm_configured(self) -> bool:
        key = self.settings.groq_api_key if self.settings.llm_provider == "groq" else self.settings.gemini_api_key
        return bool(key) or self._llm is not None

    def videos(self, playlist: str | None = None) -> list[dict]:
        built = self.available_strategies(playlist)
        if not built:
            return []
        st = self.library.settings_for(playlist)
        strategy = st.default_strategy if st.default_strategy in built else built[0]
        by_video: dict[str, dict] = {}
        for c in load_chunks(chunks_path(st, strategy)):
            v = by_video.setdefault(
                c.video_id,
                {
                    "video_id": c.video_id,
                    "title": c.title,
                    "playlist_index": c.playlist_index,
                    "duration": 0.0,
                    "chunk_count": 0,
                },
            )
            v["duration"] = max(v["duration"], c.end)
            v["chunk_count"] += 1
        return sorted(by_video.values(), key=lambda v: (v["playlist_index"], v["video_id"]))

    # -- core -------------------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        *,
        strategy: str | None = None,
        mode: str = "hybrid",
        rerank: bool = False,
        expand: bool = False,
        video_ids: Sequence[str] | None = None,
        rrf_weights: tuple[float, float] | None = None,
        playlist: str | None = None,
    ) -> SearchResult:
        """Defaults match what the eval supports: hybrid with weighted RRF (settings.rrf_weights),
        no cross-encoder rerank, no glossary expansion. Both extras are opt-in."""
        weights = rrf_weights if rrf_weights is not None else self.settings.rrf_weights
        return self.retriever(strategy, playlist).search(
            query, k, mode=mode, rerank=rerank, expand=expand, video_ids=video_ids, rrf_weights=weights
        )

    def ask(
        self,
        question: str,
        k: int = 6,
        *,
        strategy: str | None = None,
        mode: str = "hybrid",
        rerank: bool = False,
        expand: bool = False,
        video_ids: Sequence[str] | None = None,
        rrf_weights: tuple[float, float] | None = None,
        playlist: str | None = None,
    ) -> AskResult:
        llm = self.llm  # fail fast if no key configured, before spending time on retrieval
        result = self.search(
            question,
            k,
            strategy=strategy,
            mode=mode,
            rerank=rerank,
            expand=expand,
            video_ids=video_ids,
            rrf_weights=rrf_weights,
            playlist=playlist,
        )
        timings = dict(result.timings_ms)
        if not result.hits:
            return AskResult(
                "I couldn't find anything relevant in the lectures.", [], [], timings, llm.model,
                top_dense=result.top_dense,
            )
        if result.out_of_scope:  # no LLM call: nothing to ground an answer on
            return AskResult(
                OUT_OF_SCOPE_ANSWER, result.hits, [], timings, llm.model,
                result.rerank_applied, True, result.top_dense,
            )

        t = perf_counter()
        answer = llm.chat(build_messages(question, result.hits, self.library.title(playlist)))
        timings["llm"] = round((perf_counter() - t) * 1000, 1)
        if answer.strip().upper().startswith("NOT_COVERED"):  # model judged the excerpts irrelevant
            return AskResult(
                NOT_COVERED_ANSWER, result.hits, [], timings, llm.model,
                result.rerank_applied, True, result.top_dense,
            )
        return AskResult(
            answer=answer,
            hits=result.hits,
            cited=cited_indices(answer, len(result.hits)),
            timings_ms=timings,
            model=llm.model,
            rerank_applied=result.rerank_applied,
            top_dense=result.top_dense,
        )
