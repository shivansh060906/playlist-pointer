"""Timestamp-aware retrieval metrics.

Ground truth is a list of (video_id, timestamp) pairs per question: the moment(s) where the
question is actually answered. A retrieved chunk is a HIT if it is in the right video and its span,
widened by +/- `tol` seconds, contains a labelled timestamp.

Because bigger chunks trivially contain more timestamps, hit@k alone flatters big chunks. Always read it
next to `avg_chunk_s` and `median_seek_error_s` (how far from the true moment the top-1 link lands).
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .utils import parse_timestamp


@dataclass(frozen=True)
class Answer:
    video_id: str
    timestamp: float


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answers: tuple[Answer, ...]


def load_questions(path: Path) -> list[Question]:
    """Accepts a list, or {"questions": [...]}. Each item: {id?, question, video_id+timestamp | answers: [...]}.
    Items still containing the REPLACE placeholder (from the example file) are skipped."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) else raw
    questions: list[Question] = []
    for i, item in enumerate(items, start=1):
        pairs = item.get("answers") or [{"video_id": item.get("video_id"), "timestamp": item.get("timestamp")}]
        answers = tuple(
            Answer(p["video_id"], parse_timestamp(p["timestamp"]))
            for p in pairs
            if p.get("video_id") and not str(p["video_id"]).startswith("REPLACE")
        )
        if answers:
            questions.append(Question(item.get("id", f"q{i}"), item["question"], answers))
    return questions


def span_hits(video_id: str, start: float, end: float, answers: Sequence[Answer], tol: float) -> bool:
    return any(a.video_id == video_id and start - tol <= a.timestamp <= end + tol for a in answers)


def first_hit_rank(
    spans: Sequence[tuple[str, float, float]], answers: Sequence[Answer], tol: float
) -> int | None:
    """1-based rank of the first retrieved span that hits, else None."""
    for rank, (video_id, start, end) in enumerate(spans, start=1):
        if span_hits(video_id, start, end, answers, tol):
            return rank
    return None


def seek_error(span: tuple[str, float, float], answers: Sequence[Answer]) -> float | None:
    """Seconds between where the link would land (span start) and the nearest labelled moment."""
    video_id, start, _ = span
    distances = [abs(a.timestamp - start) for a in answers if a.video_id == video_id]
    return min(distances) if distances else None


def summarize(rows: Sequence[dict], ks: Sequence[int] = (1, 3, 5)) -> dict:
    """rows: one dict per question with keys rank (int|None), seek_error (float|None), latency_ms."""
    n = len(rows)
    if n == 0:
        return {}
    out: dict = {"n": n}
    for k in ks:
        out[f"hit@{k}"] = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / n
    out["mrr"] = sum(1.0 / r["rank"] for r in rows if r["rank"]) / n
    top1_errors = [r["seek_error"] for r in rows if r["rank"] == 1 and r["seek_error"] is not None]
    out["median_seek_error_s"] = statistics.median(top1_errors) if top1_errors else None
    latencies = sorted(r["latency_ms"] for r in rows)
    out["p50_latency_ms"] = latencies[len(latencies) // 2]
    return out


# ------------------------------------------------------------------ statistics
def per_question_arrays(ranks: Sequence[int | None]) -> dict[str, np.ndarray]:
    """One value per question, so configurations can be compared question-by-question (paired)."""
    return {
        "hit@1": np.array([1.0 if r is not None and r <= 1 else 0.0 for r in ranks]),
        "hit@3": np.array([1.0 if r is not None and r <= 3 else 0.0 for r in ranks]),
        "hit@5": np.array([1.0 if r is not None and r <= 5 else 0.0 for r in ranks]),
        "rr": np.array([1.0 / r if r else 0.0 for r in ranks]),
    }


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean: resample questions with replacement."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean(a - b), ci_lo, ci_hi) over the same questions. If the CI excludes 0, the two
    configurations differ by more than question-sampling noise."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d, n_boot, alpha, seed)
    return float(d.mean()), lo, hi


def excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


# ---------------------------------------------------------------- miss analysis
MISS_CATEGORIES = (
    "label: video not indexed",
    "label: timestamp past end of video",
    "hit below rank 1",
    "right video, wrong moment",
    "wrong video",
)


def _distance_to_span(t: float, start: float, end: float) -> float:
    return 0.0 if start <= t <= end else min(abs(t - start), abs(t - end))


def classify_miss(
    spans: Sequence[tuple[str, float, float]],
    answers: Sequence[Answer],
    tol: float,
    video_extent: dict[str, float],
) -> tuple[str, str]:
    """Why did this question not get a hit at rank 1?  Returns (category, human-readable detail).

    `video_extent` maps video_id -> last indexed timestamp, which lets us separate genuine retrieval
    failures from labelling mistakes (wrong id, or a timestamp beyond the end of the video).
    """
    known = [a for a in answers if a.video_id in video_extent]
    if not known:
        return MISS_CATEGORIES[0], "none of the labelled video ids exist in the index (typo or not ingested?)"
    if all(a.timestamp > video_extent[a.video_id] + tol for a in known):
        detail = ", ".join(f"{a.timestamp:.0f}s > video end {video_extent[a.video_id]:.0f}s" for a in known)
        return MISS_CATEGORIES[1], detail

    rank = first_hit_rank(spans, answers, tol)
    if rank == 1:
        return "hit at rank 1", ""
    if rank is not None:
        return MISS_CATEGORIES[2], f"first correct result is at rank {rank}"

    answer_videos = {a.video_id for a in known}
    distances = [
        _distance_to_span(a.timestamp, start, end)
        for (video_id, start, end) in spans
        for a in known
        if video_id == a.video_id
    ]
    if distances:
        return MISS_CATEGORIES[3], f"nearest retrieved chunk in the right video is {min(distances):.0f}s from the label"
    return MISS_CATEGORIES[4], f"no result from the labelled video(s) {sorted(answer_videos)} in the retrieved list"


def bootstrap_median_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the median (used for seek error, which is a median over the
    questions whose top-1 result was correct, so the sample is small and the CI is honest about it)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    medians = np.median(v[rng.integers(0, v.size, size=(n_boot, v.size))], axis=1)
    lo, hi = np.percentile(medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
