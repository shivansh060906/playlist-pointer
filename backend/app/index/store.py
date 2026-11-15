"""On-disk index for one chunking strategy: chunks (jsonl) + Chroma vectors + pickled BM25."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..config import Settings
from ..models import Chunk
from ..retrieval.tokenizer import tokenize


class IndexNotBuilt(RuntimeError):
    def __init__(self, strategy: str):
        super().__init__(
            f"Index for strategy '{strategy}' has not been built. "
            f"Run: python -m scripts.build_index --strategy {strategy}"
        )
        self.strategy = strategy


def chunks_path(settings: Settings, strategy: str) -> Path:
    return settings.index_dir / f"chunks_{strategy}.jsonl"


def bm25_path(settings: Settings, strategy: str) -> Path:
    return settings.index_dir / f"bm25_{strategy}.pkl"


def index_exists(settings: Settings, strategy: str) -> bool:
    return chunks_path(settings, strategy).exists() and bm25_path(settings, strategy).exists()


def save_chunks(path: Path, chunks: Iterable[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_json() + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk.from_json(line) for line in f if line.strip()]


class VectorStore:
    """Chroma collection (cosine space), one per chunking strategy."""

    def __init__(self, chroma_dir: Path, strategy: str):
        import chromadb

        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self.name = f"chunks_{strategy}"

    def rebuild(self, chunks: Sequence[Chunk], embeddings: np.ndarray, batch_size: int = 500) -> None:
        try:
            self._client.delete_collection(self.name)
        except Exception:  # noqa: BLE001  (collection did not exist)
            pass
        collection = self._client.create_collection(self.name, metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), batch_size):
            part = chunks[i : i + batch_size]
            collection.add(
                ids=[c.chunk_id for c in part],
                embeddings=np.asarray(embeddings[i : i + batch_size], dtype=np.float32).tolist(),
                documents=[c.text for c in part],
                metadatas=[
                    {"video_id": c.video_id, "start": float(c.start), "end": float(c.end)} for c in part
                ],
            )

    def count(self) -> int:
        return self._client.get_collection(self.name).count()

    def query(
        self, embedding: np.ndarray, n: int, video_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """Returns [(chunk_id, cosine_similarity)] best first."""
        collection = self._client.get_collection(self.name)
        total = collection.count()
        if total == 0:
            return []
        where = {"video_id": {"$in": sorted(video_ids)}} if video_ids else None
        result = collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(n, total),
            where=where,
        )
        return [(cid, 1.0 - float(dist)) for cid, dist in zip(result["ids"][0], result["distances"][0])]


class BM25Store:
    def __init__(self, bm25, chunk_ids: list[str]):
        self._bm25 = bm25
        self._ids = chunk_ids

    @classmethod
    def build(cls, chunks: Sequence[Chunk]) -> "BM25Store":
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(c.embed_text) for c in chunks]
        return cls(BM25Okapi(corpus), [c.chunk_id for c in chunks])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"bm25": self._bm25, "ids": self._ids}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Store":
        with path.open("rb") as f:  # our own file, written by build_index
            payload = pickle.load(f)
        return cls(payload["bm25"], payload["ids"])

    @property
    def doc_count(self) -> int:
        return len(self._ids)

    def _df_tokens(self):
        self.doc_freq("")  # make sure the cache is built
        return self._df.keys()

    def doc_freq(self, token: str) -> int:
        """Number of chunks containing `token` (0 if it never occurs in the corpus)."""
        df = getattr(self, "_df", None)
        if df is None:
            df = {}
            for freqs in self._bm25.doc_freqs:
                for t in freqs:
                    df[t] = df.get(t, 0) + 1
            self._df = df
        return df.get(token, 0)

    def query(
        self, tokens: list[str], n: int, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens))
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for idx in order:
            if scores[idx] <= 0:
                break
            cid = self._ids[idx]
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            out.append((cid, float(scores[idx])))
            if len(out) >= n:
                break
        return out
