"""Evaluation runner: ablation grid, bootstrap CIs, paired comparisons and a categorised miss report.

Kept in `app/` (not in the CLI script) so it can be unit-tested against the synthetic corpus.
"""
from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .evaluation import (
    MISS_CATEGORIES,
    Answer,
    Question,
    bootstrap_ci,
    bootstrap_median_ci,
    classify_miss,
    excludes_zero,
    first_hit_rank,
    paired_bootstrap,
    per_question_arrays,
    seek_error,
    summarize,
)
from .evaluation import _distance_to_span as distance_to_span
from .utils import format_timestamp

# (label, kwargs for Retriever.search). Each row isolates one idea so the table shows what it buys.
# "2:1" / "3:1" are weighted-RRF dense:bm25 weights.
CONFIGS: list[tuple[str, dict]] = [
    ("dense only", dict(mode="dense", rerank=False, expand=False)),
    ("bm25 only", dict(mode="bm25", rerank=False, expand=False)),
    ("bm25 + glossary", dict(mode="bm25", rerank=False, expand=True)),
    ("hybrid (RRF)", dict(mode="hybrid", rerank=False, expand=False)),
    ("hybrid RRF 2:1", dict(mode="hybrid", rerank=False, expand=False, rrf_weights=(2.0, 1.0))),
    ("hybrid RRF 3:1", dict(mode="hybrid", rerank=False, expand=False, rrf_weights=(3.0, 1.0))),
    ("hybrid + glossary", dict(mode="hybrid", rerank=False, expand=True)),
    ("hybrid 2:1 + glossary", dict(mode="hybrid", rerank=False, expand=True, rrf_weights=(2.0, 1.0))),
    ("hybrid + rerank", dict(mode="hybrid", rerank=True, expand=False)),
    ("hybrid + glossary + rerank", dict(mode="hybrid", rerank=True, expand=True)),
]

# Single-factor comparisons: (reference config, variant config) differ in exactly one thing, so a
# paired difference isolates that factor. (Comparing everything to "dense only" cannot say, e.g.,
# whether the glossary helps hybrid search.)
EFFECT_PAIRS: dict[str, list[tuple[str, str]]] = {
    "Glossary expansion on vs off": [
        ("bm25 only", "bm25 + glossary"),
        ("hybrid (RRF)", "hybrid + glossary"),
        ("hybrid RRF 2:1", "hybrid 2:1 + glossary"),
        ("hybrid + rerank", "hybrid + glossary + rerank"),
    ],
    "Weighted RRF vs equal-weight RRF": [
        ("hybrid (RRF)", "hybrid RRF 2:1"),
        ("hybrid (RRF)", "hybrid RRF 3:1"),
    ],
}


# ------------------------------------------------------------------------- run
def run_eval(
    retrievers: Mapping[str, object],
    questions: Sequence[Question],
    configs: Sequence[tuple[str, dict]] = CONFIGS,
    *,
    tol: float = 30.0,
    depth: int = 10,
    n_boot: int = 2000,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict:
    per_q: dict[str, dict[str, list[dict]]] = {}
    summaries: list[dict] = []

    for strategy, retriever in retrievers.items():
        per_q[strategy] = {}
        for label, cfg in configs:
            rows = []
            for q in questions:
                res = retriever.search(q.question, depth, **cfg)
                spans = [(h.chunk.video_id, h.chunk.start, h.chunk.end) for h in res.hits]
                rows.append(
                    {
                        "rank": first_hit_rank(spans, q.answers, tol),
                        "seek_error": seek_error(spans[0], q.answers) if spans else None,
                        "latency_ms": res.timings_ms["total"],
                        "spans": spans,
                        "top": [
                            {
                                "video_id": h.chunk.video_id,
                                "title": h.chunk.title,
                                "start": h.chunk.start,
                                "text": h.chunk.text,
                            }
                            for h in res.hits[:3]
                        ],
                    }
                )
            per_q[strategy][label] = rows

            arrays = per_question_arrays([r["rank"] for r in rows])
            summary = summarize(rows)
            summary["strategy"] = strategy
            summary["config"] = label
            summary["avg_chunk_s"] = round(retriever.avg_chunk_seconds, 1)
            summary["hit@1_ci"] = list(bootstrap_ci(arrays["hit@1"], n_boot, seed=seed))
            summary["mrr_ci"] = list(bootstrap_ci(arrays["rr"], n_boot, seed=seed))
            top1_errors = [r["seek_error"] for r in rows if r["rank"] == 1 and r["seek_error"] is not None]
            summary["median_seek_error_ci"] = (
                list(bootstrap_median_ci(top1_errors, n_boot, seed=seed)) if top1_errors else None
            )
            summaries.append(summary)
            lo, hi = summary["hit@1_ci"]
            log(
                f"{strategy:<9} {label:<26} hit@1 {summary['hit@1']:.2f} ({lo:.2f}-{hi:.2f})  "
                f"hit@5 {summary['hit@5']:.2f}  MRR {summary['mrr']:.3f}"
            )
    return {"summaries": summaries, "per_q": per_q}


# ---------------------------------------------------------------------- paired
def paired_comparisons(per_q: Mapping[str, Mapping[str, list[dict]]], baseline: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Question-by-question differences with bootstrap CIs.

    vs_baseline: every configuration against `baseline`, within each chunking strategy.
    vs_fixed:    every strategy against `fixed`, within each configuration.
    effects:     single-factor comparisons (EFFECT_PAIRS): glossary on/off, weighted vs equal RRF.
    """

    def arrays(rows):
        return per_question_arrays([r["rank"] for r in rows])

    def entry(cur, ref):
        out = {}
        for key, name in (("hit@1", "hit@1"), ("rr", "mrr")):
            diff, lo, hi = paired_bootstrap(cur[key], ref[key], n_boot, seed=seed)
            out[name] = {"diff": diff, "ci": [lo, hi], "significant": excludes_zero(lo, hi)}
        return out

    vs_baseline, vs_fixed = [], []
    for strategy, by_cfg in per_q.items():
        if baseline in by_cfg:
            ref = arrays(by_cfg[baseline])
            for label, rows in by_cfg.items():
                if label != baseline:
                    vs_baseline.append(
                        {"strategy": strategy, "config": label, "reference": baseline, **entry(arrays(rows), ref)}
                    )
    if "fixed" in per_q:
        for strategy, by_cfg in per_q.items():
            if strategy == "fixed":
                continue
            for label, rows in by_cfg.items():
                if label in per_q["fixed"]:
                    vs_fixed.append(
                        {
                            "strategy": strategy,
                            "config": label,
                            "reference": "fixed",
                            **entry(arrays(rows), arrays(per_q["fixed"][label])),
                        }
                    )
    effects = []
    for effect, pairs in EFFECT_PAIRS.items():
        for strategy, by_cfg in per_q.items():
            for ref_label, variant_label in pairs:
                if ref_label in by_cfg and variant_label in by_cfg:
                    effects.append(
                        {
                            "effect": effect,
                            "strategy": strategy,
                            "config": variant_label,
                            "reference": ref_label,
                            **entry(arrays(by_cfg[variant_label]), arrays(by_cfg[ref_label])),
                        }
                    )
    return {"baseline": baseline, "vs_baseline": vs_baseline, "vs_fixed": vs_fixed, "effects": effects}


# ---------------------------------------------------------------------- misses
def _text_at(chunks: Sequence, answer: Answer) -> str:
    """What the index actually holds at the labelled moment (to spot vocabulary mismatches)."""
    in_video = [c for c in chunks if c.video_id == answer.video_id]
    if not in_video:
        return ""
    containing = [c for c in in_video if c.start <= answer.timestamp <= c.end]
    if containing:
        return min(containing, key=lambda c: c.duration).text
    return min(in_video, key=lambda c: distance_to_span(answer.timestamp, c.start, c.end)).text


def collect_misses(retriever, questions: Sequence[Question], rows: Sequence[dict], tol: float) -> dict:
    extent: dict[str, float] = {}
    titles: dict[str, str] = {}
    for c in retriever.chunks:
        extent[c.video_id] = max(extent.get(c.video_id, 0.0), c.end)
        titles[c.video_id] = c.title

    misses = []
    for q, row in zip(questions, rows):
        if row["rank"] == 1:
            continue
        category, detail = classify_miss(row["spans"], q.answers, tol, extent)
        misses.append(
            {
                "id": q.id,
                "question": q.question,
                "category": category,
                "detail": detail,
                "rank": row["rank"],
                "expected": [
                    {
                        "video_id": a.video_id,
                        "title": titles.get(a.video_id, "(not in index)"),
                        "timestamp": format_timestamp(a.timestamp),
                        "text_at_label": _text_at(retriever.chunks, a),
                    }
                    for a in q.answers
                ],
                "retrieved": [
                    {
                        "rank": i,
                        "video_id": t["video_id"],
                        "title": t["title"],
                        "timestamp": format_timestamp(t["start"]),
                        "text": t["text"],
                    }
                    for i, t in enumerate(row["top"], start=1)
                ],
            }
        )
    order = {name: i for i, name in enumerate(MISS_CATEGORIES)}
    misses.sort(key=lambda m: order.get(m["category"], 99))  # label problems first: they are the cheapest fixes
    return {
        "total_questions": len(questions),
        "counts": dict(Counter(m["category"] for m in misses)),
        "misses": misses,
    }


def _snip(text: str, width: int = 260) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" ...")


def render_misses_md(report: dict, strategy: str, config: str) -> str:
    misses = report["misses"]
    lines = [
        f"# Misses at hit@1: {strategy} / {config}",
        "",
        f"{len(misses)} of {report['total_questions']} questions had no correct result at rank 1.",
        "",
        "| category | count |",
        "|---|---|",
    ]
    for category in MISS_CATEGORIES:
        if report["counts"].get(category):
            lines.append(f"| {category} | {report['counts'][category]} |")
    lines += [
        "",
        "`label:` categories are mistakes in questions.json (fix those first); the rest are retrieval failures.",
        "",
    ]
    for m in misses:
        lines += [f"## {m['id']} ({m['category']})", "", f"**Question:** {m['question']}", ""]
        if m["detail"]:
            lines += [f"*{m['detail']}*", ""]
        for e in m["expected"]:
            lines.append(f"**Expected:** {e['title']} (`{e['video_id']}`) @ {e['timestamp']}")
            if e["text_at_label"]:
                lines += ["", f"> {_snip(e['text_at_label'])}"]
            lines.append("")
        lines.append("**Retrieved:**")
        lines.append("")
        for r in m["retrieved"]:
            lines.append(f"{r['rank']}. {r['title']} (`{r['video_id']}`) @ {r['timestamp']}: \"{_snip(r['text'], 200)}\"")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- reports
def _fmt_diff(e: dict) -> str:
    d, (lo, hi) = e["diff"], e["ci"]
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if e['significant'] else ''}"


def _fmt_ci(value: float, ci: Sequence[float], digits: int) -> str:
    return f"{value:.{digits}f} ({ci[0]:.{digits}f}-{ci[1]:.{digits}f})"


def _paired_table(title: str, note: str, rows: list[dict], show_reference: bool = False) -> list[str]:
    lines = [f"## {title}", "", note, ""]
    if not rows:
        return lines + ["(nothing to compare)", ""]
    lines += ["| chunking | configuration | d hit@1 | d MRR |", "|---|---|---|---|"]
    for e in rows:
        label = f"{e['config']} vs {e['reference']}" if show_reference else e["config"]
        lines.append(f"| {e['strategy']} | {label} | {_fmt_diff(e['hit@1'])} | {_fmt_diff(e['mrr'])} |")
    return lines + [""]


def write_reports(
    out_dir: Path,
    result: dict,
    paired: dict,
    misses: dict | None,
    *,
    n_questions: int,
    tol: float,
    n_boot: int,
    misses_target: tuple[str, str] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = result["summaries"]

    (out_dir / "results.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (out_dir / "paired.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    light = {
        strategy: {
            label: [{"rank": r["rank"], "seek_error": r["seek_error"]} for r in rows]
            for label, rows in by_cfg.items()
        }
        for strategy, by_cfg in result["per_q"].items()
    }
    (out_dir / "per_question.json").write_text(json.dumps(light), encoding="utf-8")

    lines = [
        f"Retrieval eval: {n_questions} questions, tolerance +/-{tol:g}s, {n_boot} bootstrap resamples.",
        "",
        "Intervals are 95% bootstrap CIs over questions. When two intervals overlap heavily, treat the "
        "configurations as indistinguishable; see the paired tables below for direct comparisons.",
        "",
        "| chunking | configuration | hit@1 (95% CI) | hit@3 | hit@5 | MRR (95% CI) | median seek err (s, 95% CI) | avg chunk (s) | p50 latency (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summaries:
        if r["median_seek_error_s"] is None:
            seek = "-"
        else:
            seek = f"{r['median_seek_error_s']:.0f}"
            ci = r.get("median_seek_error_ci")
            if ci:
                seek += f" ({ci[0]:.0f}-{ci[1]:.0f})"
        lines.append(
            f"| {r['strategy']} | {r['config']} | {_fmt_ci(r['hit@1'], r['hit@1_ci'], 2)} | {r['hit@3']:.2f} | "
            f"{r['hit@5']:.2f} | {_fmt_ci(r['mrr'], r['mrr_ci'], 3)} | {seek} | {r['avg_chunk_s']:.0f} | "
            f"{r['p50_latency_ms']:.0f} |"
        )
    lines.append("")
    note = (
        "Paired bootstrap on per-question differences, 95% CI; `*` = CI excludes 0. "
        "With this many comparisons, roughly 1 in 20 starred results can be chance."
    )
    lines += _paired_table(
        f"Configurations vs \"{paired['baseline']}\" (same chunking)", "d = configuration minus baseline. " + note, paired["vs_baseline"]
    )
    lines += _paired_table(
        "Chunking strategies vs fixed (same configuration)", "d = strategy minus fixed. " + note, paired["vs_fixed"]
    )
    for effect in EFFECT_PAIRS:
        rows = [e for e in paired.get("effects", []) if e["effect"] == effect]
        lines += _paired_table(
            effect, "d = variant minus reference; the two differ in only this one factor. " + note, rows,
            show_reference=True,
        )
    lines.append(
        "Seek error is the distance from where the top-1 link lands to the labelled moment, over the questions "
        "whose top-1 result was correct (so the sample differs per row; read its interval as wide)."
    )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if misses is not None and misses_target is not None:
        (out_dir / "misses.json").write_text(json.dumps(misses, indent=2), encoding="utf-8")
        (out_dir / "misses.md").write_text(render_misses_md(misses, *misses_target), encoding="utf-8")


def significant_lines(paired: dict) -> list[str]:
    out = []
    for section in ("vs_baseline", "vs_fixed", "effects"):
        for e in paired.get(section, []):
            for metric in ("hit@1", "mrr"):
                if e[metric]["significant"]:
                    out.append(
                        f"  {e['strategy']:<9} {e['config']:<26} vs {e['reference']:<11} {metric:<5} {_fmt_diff(e[metric])}"
                    )
    return out


def evaluate_and_report(
    retrievers: Mapping[str, object],
    questions: Sequence[Question],
    out_dir: Path,
    *,
    configs: Sequence[tuple[str, dict]] = CONFIGS,
    baseline: str = "dense only",
    tol: float = 30.0,
    depth: int = 10,
    n_boot: int = 2000,
    seed: int = 0,
    misses_strategy: str | None = "semantic",
    misses_config: str = "hybrid + glossary",
    log: Callable[[str], None] = print,
) -> dict:
    labels = [label for label, _ in configs]
    if baseline not in labels:
        raise ValueError(f"baseline {baseline!r} is not one of the configurations: {labels}")

    result = run_eval(retrievers, questions, configs, tol=tol, depth=depth, n_boot=n_boot, seed=seed, log=log)
    paired = paired_comparisons(result["per_q"], baseline, n_boot, seed)

    misses, target = None, None
    if misses_strategy:
        if misses_strategy in retrievers and misses_config in labels:
            target = (misses_strategy, misses_config)
            misses = collect_misses(
                retrievers[misses_strategy], questions, result["per_q"][misses_strategy][misses_config], tol
            )
        else:
            log(f"(skipping misses report: {misses_strategy!r} / {misses_config!r} was not part of this run)")

    write_reports(
        out_dir, result, paired, misses,
        n_questions=len(questions), tol=tol, n_boot=n_boot, misses_target=target,
    )

    sig = significant_lines(paired)
    log("\nDifferences whose 95% CI excludes zero (paired bootstrap):")
    log("\n".join(sig) if sig else "  none. With this many questions, no configuration is demonstrably better than another.")
    if misses is not None:
        log(f"\nMisses at hit@1 for {target[0]} / {target[1]}: {len(misses['misses'])} of {len(questions)}")
        for category, count in misses["counts"].items():
            log(f"  {count:>3}  {category}")
    log(f"\nWrote results.md, results.json, paired.json, per_question.json"
        + (", misses.md, misses.json" if misses is not None else "") + f" to {out_dir}")
    return {"result": result, "paired": paired, "misses": misses}
