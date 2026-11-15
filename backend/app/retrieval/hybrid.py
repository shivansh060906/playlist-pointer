"""Hybrid retriever: dense (Chroma) + BM25 -> RRF fusion -> optional cross-encoder rerank -> dedupe."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

from ..config import Settings
from ..index.store import (
    BM25Store,
    IndexNotBuilt,
    VectorStore,
    bm25_path,
    chunks_path,
    index_exists,
    load_chunks,
)
from ..models import Chunk
from .fusion import reciprocal_rank_fusion
from .glossary import Glossary
from .tokenizer import tokenize

MODES = ("hybrid", "dense", "bm25")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    dense_score: float | None = None  # cosine similarity to the query (None if not in the dense list)


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]
    mode: str
    rerank_applied: bool
    expanded_query: str | None = None
    top_dense: float | None = None  # best cosine similarity among dense candidates
    out_of_scope: bool = False  # query looks unrelated to the indexed lectures


def _ms(since: float) -> float:
    return round((perf_counter() - since) * 1000, 1)


def overlap_ratio(a: Chunk, b: Chunk) -> float:
    """Temporal overlap of two chunks of the same video, relative to the shorter one."""
    if a.video_id != b.video_id:
        return 0.0
    overlap = min(a.end, b.end) - max(a.start, b.start)
    shortest = min(a.duration, b.duration)
    return max(0.0, overlap) / shortest if shortest > 0 else 0.0


def dedupe_overlaps(hits: Sequence[Hit], threshold: float = 0.5) -> list[Hit]:
    """Drop lower-ranked hits that mostly cover the same moment as a better one
    (sliding windows produce many near-identical neighbours)."""
    kept: list[Hit] = []
    for hit in hits:
        if all(overlap_ratio(hit.chunk, k.chunk) < threshold for k in kept):
            kept.append(hit)
    return kept


@dataclass
class VideoGroup:
    video_id: str
    title: str
    playlist_index: int
    best_score: float
    hit_count: int
    best_chunk: Chunk = field(repr=False, default=None)  # type: ignore[assignment]


def group_hits_by_video(hits: Sequence[Hit]) -> list[VideoGroup]:
    """'Which episodes cover X?' -- hits are already best-first, so first sighting = best chunk."""
    groups: dict[str, VideoGroup] = {}
    for hit in hits:
        c = hit.chunk
        if c.video_id not in groups:
            groups[c.video_id] = VideoGroup(c.video_id, c.title, c.playlist_index, hit.score, 1, c)
        else:
            groups[c.video_id].hit_count += 1
    return list(groups.values())


class Retriever:
    def __init__(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: VectorStore,
        bm25: BM25Store,
        embedder,
        reranker=None,
        glossary: Glossary | None = None,
        min_relevance: float = 0.0,
        rare_df_fraction: float = 0.01,
    ):
        self.min_relevance = min_relevance  # 0 disables the out-of-scope check
        self.rare_df_fraction = rare_df_fraction
        self.strategy = strategy
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self.vectors = vectors
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.glossary = glossary

    @classmethod
    def load(cls, settings: Settings, strategy: str, embedder, reranker=None, glossary=None):
        if not index_exists(settings, strategy):
            raise IndexNotBuilt(strategy)
        return cls(
            strategy=strategy,
            chunks=load_chunks(chunks_path(settings, strategy)),
            vectors=VectorStore(settings.chroma_dir, strategy),
            bm25=BM25Store.load(bm25_path(settings, strategy)),
            embedder=embedder,
            reranker=reranker,
            glossary=glossary,
            min_relevance=settings.min_relevance,
        )

    def has_rare_term(self, query: str) -> bool:
        """True if the query contains a term that is specific to a few chunks (e.g. `LNK2005`,
        `std::move`). Such queries are legitimate even when the embedding similarity is low."""
        cap = max(3, int(self.rare_df_fraction * self.bm25.doc_count))
        return any(0 < self.bm25.doc_freq(t) <= cap for t in set(tokenize(query)))

    def is_out_of_scope(self, query: str, top_dense: float | None) -> bool:
        if self.min_relevance <= 0 or top_dense is None:
            return False
        return top_dense < self.min_relevance and not self.has_rare_term(query)

    @property
    def avg_chunk_seconds(self) -> float:
        return sum(c.duration for c in self.chunks) / len(self.chunks) if self.chunks else 0.0

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        mode: str = "hybrid",
        rerank: bool = False,
        expand: bool = False,
        candidates: int = 30,
        video_ids: Sequence[str] | None = None,
        dedupe: bool = True,
        rrf_weights: tuple[float, float] | None = None,
    ) -> SearchResult:
        """`rrf_weights` = (dense_weight, bm25_weight) for hybrid fusion; None means equal weights."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if rrf_weights is not None and len(rrf_weights) != 2:
            raise ValueError("rrf_weights must be (dense_weight, bm25_weight)")

        started = perf_counter()
        timings: dict[str, float] = {}
        video_filter = set(video_ids) if video_ids else None

        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []
        expanded: str | None = None

        if mode in ("dense", "hybrid"):
            t = perf_counter()
            dense = self.vectors.query(self.embedder.embed_query(query), candidates, video_filter)
            timings["dense"] = _ms(t)

        if mode in ("bm25", "hybrid"):
            t = perf_counter()
            bm25_query = query
            if expand and self.glossary is not None:
                bm25_query = self.glossary.expand(query)
                expanded = bm25_query if bm25_query != query else None
            allowed = (
                {c.chunk_id for c in self.chunks if c.video_id in video_filter} if video_filter else None
            )
            sparse = self.bm25.query(tokenize(bm25_query), candidates, allowed)
            timings["bm25"] = _ms(t)

        dense_score = {cid: sc for cid, sc in dense}
        top_dense = dense[0][1] if dense else None
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense, start=1)}
        sparse_rank = {cid: r for r, (cid, _) in enumerate(sparse, start=1)}

        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [[c for c, _ in dense], [c for c, _ in sparse]], weights=rrf_weights
            )
        elif mode == "dense":
            fused = dense
        else:
            fused = sparse
        ordered = fused[:candidates]

        rerank_scores: dict[str, float] = {}
        reranked = False
        if rerank and self.reranker is not None and ordered:
            t = perf_counter()
            passages = [self._by_id[cid].embed_text for cid, _ in ordered]
            scores = self.reranker.score(query, passages)
            rerank_scores = {cid: s for (cid, _), s in zip(ordered, scores)}
            ordered = sorted(ordered, key=lambda item: -rerank_scores[item[0]])
            reranked = True
            timings["rerank"] = _ms(t)

        hits = [
            Hit(
                chunk=self._by_id[cid],
                score=float(rerank_scores.get(cid, base)),
                dense_rank=dense_rank.get(cid),
                bm25_rank=sparse_rank.get(cid),
                rerank_score=rerank_scores.get(cid),
                dense_score=dense_score.get(cid),
            )
            for cid, base in ordered
        ]
        if dedupe:
            hits = dedupe_overlaps(hits)
        timings["total"] = _ms(started)
        return SearchResult(
            hits[:k], timings, mode, reranked, expanded, top_dense, self.is_out_of_scope(query, top_dense)
        )
