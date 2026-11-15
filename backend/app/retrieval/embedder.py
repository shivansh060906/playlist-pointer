from __future__ import annotations

import threading
from typing import Sequence

import numpy as np


def _prefixes_for(model_name: str) -> tuple[str, str]:
    """(query_prefix, document_prefix). Retrieval models expect these asymmetric instructions."""
    name = model_name.lower()
    if "bge" in name and "m3" not in name:
        return "Represent this sentence for searching relevant passages: ", ""
    if "e5" in name:
        return "query: ", "passage: "
    return "", ""


class SentenceTransformerEmbedder:
    """Thin wrapper: lazy model load, L2-normalised float32 vectors, query/document prefixes."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.doc_prefix = _prefixes_for(model_name)
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        vectors = self._load().encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), self.doc_prefix)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], self.query_prefix)[0]
