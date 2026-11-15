"""Calibration helpers for the out-of-scope guard (see scripts/calibrate.py)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Queries that a C++ lecture search should refuse. Deliberately a mix of obvious nonsense and
# "plausible but not C++" so the threshold isn't tuned only against absurd inputs.
OFF_TOPIC_QUERIES = [
    "tell me more about my little pony",
    "what is the best pizza recipe",
    "who won the world cup in 2018",
    "how do I lose weight fast",
    "what's the weather like in london",
    "explain the plot of the lord of the rings",
    "how do I bake sourdough bread",
    "best budget smartphone to buy",
    "how to train a puppy to sit",
    "what is the capital of australia",
    "tips for a job interview",
    "how do vaccines work",
    "who is taylor swift dating",
    "how do I center a div in css",
    "what is a python list comprehension",
    "how to set up a react useEffect hook",
    "explain photosynthesis",
    "what is the meaning of life",
    "how do I change a car tyre",
    "history of the roman empire",
    "write me a poem about the sea",
    "how much does a tesla cost",
    "what is bitcoin",
    "how to play chess openings",
    "asdf qwerty zxcv",
    "hello how are you",
    "recommend a good anime",
    "how do I make a cup of tea",
    "what does a kubernetes pod do",
    "explain the French revolution",
]


@dataclass
class Calibration:
    threshold: float
    in_scope_refused: float  # fraction of in-scope queries wrongly refused
    off_topic_refused: float  # fraction of off-topic queries correctly refused
    n_in: int
    n_off: int


def refusal_rate(scores: Sequence[float], rare: Sequence[bool], threshold: float) -> float:
    if not scores:
        return 0.0
    return sum(1 for s, r in zip(scores, rare) if s < threshold and not r) / len(scores)


def suggest_threshold(
    in_scores: Sequence[float],
    off_scores: Sequence[float],
    in_rare: Sequence[bool] | None = None,
    off_rare: Sequence[bool] | None = None,
    max_false_refusal: float = 0.02,
) -> Calibration:
    """Highest-recall threshold whose false-refusal rate on in-scope questions stays within budget.

    The guard refuses when score < threshold and the query has no rare in-corpus term, so queries
    flagged rare never count toward refusals."""
    in_rare = list(in_rare) if in_rare is not None else [False] * len(in_scores)
    off_rare = list(off_rare) if off_rare is not None else [False] * len(off_scores)
    allscores = sorted(set(in_scores) | set(off_scores))
    if not allscores:
        raise ValueError("need at least one score")
    # candidate thresholds: midpoints between neighbouring observed scores, plus the extremes
    cands = [allscores[0] - 1e-3] + [(a + b) / 2 for a, b in zip(allscores, allscores[1:])] + [allscores[-1] + 1e-3]
    best: tuple[float, float, float] | None = None
    for t in cands:
        fr = refusal_rate(in_scores, in_rare, t)
        if fr > max_false_refusal:
            continue
        orr = refusal_rate(off_scores, off_rare, t)
        if best is None or orr > best[1] or (orr == best[1] and t > best[0]):
            best = (t, orr, fr)
    assert best is not None  # the lowest candidate refuses nothing, so it is always feasible
    t, orr, fr = best
    return Calibration(round(t, 4), fr, orr, len(in_scores), len(off_scores))
