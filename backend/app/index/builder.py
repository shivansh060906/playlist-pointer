from __future__ import annotations

import time
from typing import Callable

from ..chunking import chunk_transcript
from ..config import Settings
from ..models import load_all_transcripts
from .store import BM25Store, VectorStore, bm25_path, chunks_path, save_chunks


def build_index(
    settings: Settings,
    strategy: str,
    embedder,
    *,
    chunker_kwargs: dict | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Chunk every saved transcript with `strategy`, then write chunks + vectors + BM25."""
    started = time.perf_counter()
    transcripts = load_all_transcripts(settings.transcripts_dir)
    if not transcripts:
        raise FileNotFoundError(
            f"No transcripts in {settings.transcripts_dir}. Run: python -m scripts.ingest"
        )

    chunks = []
    for i, transcript in enumerate(transcripts, start=1):
        chunks.extend(
            chunk_transcript(
                transcript, strategy, embed_fn=embedder.embed_documents, **(chunker_kwargs or {})
            )
        )
        if strategy == "semantic" and i % 10 == 0:
            log(f"  chunked {i}/{len(transcripts)} videos")

    log(f"[{strategy}] {len(chunks)} chunks from {len(transcripts)} videos; embedding...")
    embeddings = embedder.embed_documents([c.embed_text for c in chunks])

    VectorStore(settings.chroma_dir, strategy).rebuild(chunks, embeddings)
    BM25Store.build(chunks).save(bm25_path(settings, strategy))
    save_chunks(chunks_path(settings, strategy), chunks)

    durations = [c.duration for c in chunks]
    stats = {
        "strategy": strategy,
        "videos": len(transcripts),
        "chunks": len(chunks),
        "avg_words": round(sum(c.word_count for c in chunks) / len(chunks), 1),
        "avg_seconds": round(sum(durations) / len(durations), 1),
        "seconds_to_build": round(time.perf_counter() - started, 1),
    }
    log(f"[{strategy}] done: {stats}")
    return stats
