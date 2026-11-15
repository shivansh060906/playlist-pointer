"""Deterministic stand-ins so the whole pipeline can be tested offline (no torch, no model downloads)."""
from __future__ import annotations

import random
import zlib

import numpy as np

from app.models import Segment, Transcript
from app.retrieval.tokenizer import tokenize


class HashingEmbedder:
    """Bag-of-words hashing embedder: texts sharing vocabulary get similar vectors."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    """Scores by query-term overlap -- enough to prove the rerank stage reorders candidates."""

    def score(self, query, passages):
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(p)))) for p in passages]


TOPICS = {
    "pointers": "pointer address memory dereference nullptr variable integer asterisk heap stack raw".split(),
    "templates": "template typename generic compile instantiation specialization parameter type function".split(),
    "linker": "linker object symbol unresolved external translation unit compile header declaration".split(),
}


def make_segments(topic_plan, seed=0, words_per_seg=8, seg_seconds=4.0):
    """topic_plan: list of (topic, n_words). Returns (segments, per-segment topic labels)."""
    rng = random.Random(seed)
    segments, labels = [], []
    t = 0.0
    for topic, n_words in topic_plan:
        vocab = TOPICS[topic]
        for _ in range(n_words // words_per_seg):
            text = " ".join(rng.choice(vocab) for _ in range(words_per_seg))
            segments.append(Segment(text=text, start=t, duration=seg_seconds))
            labels.append(topic)
            t += seg_seconds
    return segments, labels


def make_transcript(video_id, title, segments, index=1):
    return Transcript(video_id=video_id, title=title, playlist_index=index, source="test", segments=segments)
