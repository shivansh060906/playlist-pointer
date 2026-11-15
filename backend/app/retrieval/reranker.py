from __future__ import annotations

import threading
from typing import Protocol, Sequence


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Scores (query, passage) pairs jointly. Slower than bi-encoder retrieval but far more precise,
    so it is only run on the ~30 fused candidates, never on the whole corpus."""

    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._load().predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
