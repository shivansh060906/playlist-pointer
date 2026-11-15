"""Pooled relevance audit: find correct answers your original labels missed.

Why: a single labelled timestamp per question under-counts. The same question is often answered well in
more than one place (an intro and a recap, two videos, "what is X" and "how do I use X"). Counting only
the labelled moment turns valid retrievals into "misses".

How (standard pooling, as in TREC):
  1. build   - for every question, pool the top-N results from EVERY configuration, drop anything that
               already counts as correct, and merge passages covering the same moment.
  2. judge   - you answer y/n for each pooled passage. Passages are shuffled and shown without saying
               which configuration or rank produced them, so the judgments cannot favour any system.
  3. apply   - every passage you marked "y" becomes an extra accepted answer (its midpoint timestamp).

Then run the eval on both the original and the audited question file and report both.
Pooling every configuration (not just the best one) is what keeps this fair.
"""
from __future__ import annotations

import hashlib
import json
import random
import textwrap
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .evaluation import Question, span_hits
from .models import Chunk
from .retrieval.hybrid import overlap_ratio
from .utils import format_timestamp, youtube_url


def file_sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------------ pooling
def pool_candidates(
    retrievers: Mapping[str, object],
    questions: Sequence[Question],
    configs: Sequence[tuple[str, dict]],
    *,
    depth: int = 3,
    tol: float = 30.0,
    overlap: float = 0.5,
) -> list[dict]:
    """Candidate passages that some configuration ranked in its top `depth` but that the original
    labels do not already accept. Passages overlapping >= `overlap` in time are merged into one."""
    candidates: list[dict] = []
    for qi, q in enumerate(questions):
        pooled: list[tuple[Chunk, dict]] = []
        for strategy, retriever in retrievers.items():
            for label, cfg in configs:
                source = f"{strategy}/{label}"
                for hit in retriever.search(q.question, depth, **cfg).hits[:depth]:
                    chunk = hit.chunk
                    if span_hits(chunk.video_id, chunk.start, chunk.end, q.answers, tol):
                        continue  # already counted as correct under the original labels
                    existing = next((c for ch, c in pooled if overlap_ratio(chunk, ch) >= overlap), None)
                    if existing is not None:
                        if source not in existing["found_by"]:
                            existing["found_by"].append(source)
                        continue
                    cand = {
                        "key": f"{qi}:{chunk.chunk_id}",
                        "q_index": qi,
                        "question_id": q.id,
                        "question": q.question,
                        "video_id": chunk.video_id,
                        "title": chunk.title,
                        "start": chunk.start,
                        "end": chunk.end,
                        "text": chunk.text,
                        "found_by": [source],  # stored for later analysis; never shown while judging
                    }
                    pooled.append((chunk, cand))
                    candidates.append(cand)
    return candidates


# ---------------------------------------------------------------------- storage
def save_candidates(path: Path, candidates: list[dict], *, questions_sha: str, depth: int, tol: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"questions_sha": questions_sha, "depth": depth, "tol": tol, "candidates": candidates}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_candidates(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_judgments(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_judgments(path: Path, judgments: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(judgments, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- judging
def _render(c: dict, i: int, total: int) -> str:
    seek = max(0, int(c["start"]) - 2)
    excerpt = " ".join(c["text"].split())
    if len(excerpt) > 1600:
        excerpt = excerpt[:1600] + " ..."
    return "\n".join(
        [
            "-" * 78,
            f"[{i}/{total}]  QUESTION: {c['question']}",
            f"VIDEO:    {c['title']}  ({format_timestamp(c['start'])}-{format_timestamp(c['end'])})",
            f"LINK:     {youtube_url(c['video_id'], seek)}",
            "EXCERPT:",
            textwrap.fill(excerpt, width=100, initial_indent="  ", subsequent_indent="  "),
        ]
    )


def run_interactive(
    candidates: Sequence[dict],
    judgments_path: Path,
    *,
    seed: int = 0,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> dict[str, str]:
    """Ask y/n for each unjudged candidate, saving after every answer (safe to quit and resume)."""
    judgments = load_judgments(judgments_path)
    todo = [c for c in candidates if c["key"] not in judgments]
    random.Random(seed).shuffle(todo)  # order carries no information about rank or configuration
    total = len(todo)
    for i, cand in enumerate(todo, start=1):
        print_fn(_render(cand, i, total))
        while True:
            try:
                answer = input_fn("Does this excerpt, on its own, answer the question?  [y]es / [n]o / [s]kip / [q]uit > ")
            except EOFError:
                answer = "q"
            answer = answer.strip().lower()
            if answer in ("y", "n"):
                judgments[cand["key"]] = answer
                save_judgments(judgments_path, judgments)
                break
            if answer == "s":
                break
            if answer == "q":
                save_judgments(judgments_path, judgments)
                return judgments
            print_fn("please type y, n, s or q")
    return judgments


def status(candidates: Sequence[dict], judgments: Mapping[str, str]) -> dict:
    keys = {c["key"] for c in candidates}
    relevant = {k: v for k, v in judgments.items() if k in keys}
    counts = Counter(relevant.values())
    return {
        "candidates": len(keys),
        "judged": len(relevant),
        "yes": counts.get("y", 0),
        "no": counts.get("n", 0),
        "unjudged": len(keys) - len(relevant),
    }


# ------------------------------------------------------------------------ apply
def apply_judgments(
    questions: Sequence[Question], candidates: Sequence[dict], judgments: Mapping[str, str], out_path: Path
) -> dict:
    """Write an audited question file: original answers plus the midpoint of every passage judged 'y'."""
    added: dict[int, list[dict]] = {}
    for c in candidates:
        if judgments.get(c["key"]) == "y":
            midpoint = round((c["start"] + c["end"]) / 2, 1)
            added.setdefault(c["q_index"], []).append(
                {"video_id": c["video_id"], "timestamp": midpoint, "added_by": "audit"}
            )

    items = []
    for qi, q in enumerate(questions):
        answers = [{"video_id": a.video_id, "timestamp": a.timestamp} for a in q.answers]
        items.append({"id": q.id, "question": q.question, "answers": answers + added.get(qi, [])})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "_comment": "Generated by scripts.audit apply: original labels plus passages judged relevant "
                "(added_by=audit, timestamp = passage midpoint). Report results on BOTH this and the original file.",
                "questions": items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    s = status(candidates, judgments)
    return {
        "questions_changed": len(added),
        "moments_added": sum(len(v) for v in added.values()),
        "judged_yes": s["yes"],
        "judged_no": s["no"],
        "unjudged": s["unjudged"],
    }
