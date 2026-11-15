from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Merge several ranked id lists: score(id) = sum_r w_r / (k + rank_r(id)).

    RRF only uses ranks, so it needs no score normalisation between BM25 (unbounded) and cosine
    similarity ([-1, 1]), which is why it is the standard choice for hybrid search.
    """
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    scores: dict[str, float] = defaultdict(float)
    for weight, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            scores[item] += weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
