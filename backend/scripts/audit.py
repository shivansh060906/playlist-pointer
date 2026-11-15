"""Pooled relevance audit: find correct answers your original labels missed.

    python -m scripts.audit build        # pool top results from every configuration (not yet judged)
    python -m scripts.audit judge        # y/n per passage; shuffled and config-blind; resumable
    python -m scripts.audit status
    python -m scripts.audit apply        # writes eval/questions.audited.json

    python -m scripts.eval --questions eval/questions.audited.json --out eval/results_audited

Report results on BOTH the original and the audited labels, and say how many alternatives were added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.audit import (
    apply_judgments,
    file_sha,
    load_candidates,
    load_judgments,
    pool_candidates,
    run_interactive,
    save_candidates,
    status,
)
from app.config import BACKEND_DIR, STRATEGIES, get_settings
from app.engine import RAGEngine
from app.eval_runner import CONFIGS
from app.evaluation import load_questions

EVAL_DIR = BACKEND_DIR / "eval"


def _load(args) -> dict:
    path = args.audit_dir / "candidates.json"
    if not path.exists():
        raise SystemExit("No candidates yet. Run: python -m scripts.audit build")
    data = load_candidates(path)
    if args.questions.exists() and data["questions_sha"] != file_sha(args.questions):
        raise SystemExit(
            f"{args.questions} changed since `audit build`, so question numbering no longer matches. "
            "Re-run `python -m scripts.audit build` (existing judgments for unchanged passages are kept)."
        )
    return data


def cmd_build(args) -> None:
    if not args.questions.exists():
        raise SystemExit(f"{args.questions} not found.")
    questions = load_questions(args.questions)
    engine = RAGEngine(get_settings())
    strategies = args.strategies.split(",") if args.strategies else engine.available_strategies()
    for s in strategies:
        if s not in STRATEGIES:
            raise SystemExit(f"unknown strategy {s!r}")
    if not strategies:
        raise SystemExit("No indexes built. Run: python -m scripts.build_index --strategy all")

    configs = [c for c in CONFIGS if args.include_rerank or not c[1].get("rerank")]
    print(f"Pooling top {args.depth} from {len(strategies)} strategies x {len(configs)} configurations "
          f"for {len(questions)} questions...")
    candidates = pool_candidates(
        {s: engine.retriever(s) for s in strategies}, questions, configs, depth=args.depth, tol=args.tol
    )
    save_candidates(
        args.audit_dir / "candidates.json", candidates,
        questions_sha=file_sha(args.questions), depth=args.depth, tol=args.tol,
    )
    already = status(candidates, load_judgments(args.audit_dir / "judgments.json"))["judged"]
    print(f"{len(candidates)} passages to judge ({already} already judged). Next: python -m scripts.audit judge")
    if len(candidates) > 200:
        print("That is a lot to read. Consider --depth 2, or --strategies semantic.")


def cmd_judge(args) -> None:
    data = _load(args)
    print(
        "For each excerpt, decide whether it BY ITSELF would be a good answer to the question. It does not "
        "need to be the best passage, just a genuinely good one. You are not told which system found it.\n"
        "Open the link if you need context. Progress is saved after every answer; press q to stop and resume later.\n"
    )
    run_interactive(data["candidates"], args.audit_dir / "judgments.json", seed=args.seed)
    cmd_status(args, data)


def cmd_status(args, data: dict | None = None) -> None:
    data = data or _load(args)
    s = status(data["candidates"], load_judgments(args.audit_dir / "judgments.json"))
    print(f"\n{s['judged']}/{s['candidates']} judged: {s['yes']} relevant, {s['no']} not relevant, {s['unjudged']} remaining")


def cmd_apply(args) -> None:
    data = _load(args)
    questions = load_questions(args.questions)
    judgments = load_judgments(args.audit_dir / "judgments.json")
    if not judgments:
        raise SystemExit("No judgments yet. Run: python -m scripts.audit judge")
    stats = apply_judgments(questions, data["candidates"], judgments, args.out)
    print(
        f"Wrote {args.out}\n"
        f"  {stats['moments_added']} extra accepted moments across {stats['questions_changed']} questions "
        f"({stats['judged_yes']} relevant / {stats['judged_no']} not relevant / {stats['unjudged']} unjudged)"
    )
    if stats["unjudged"]:
        print("  Note: unjudged passages are treated as NOT relevant. Finish judging for a fair comparison.")
    print(f"\nNext: python -m scripts.eval --questions {args.out} --out {EVAL_DIR / 'results_audited'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json")
    common.add_argument("--audit-dir", type=Path, default=EVAL_DIR / "audit")

    b = sub.add_parser("build", parents=[common], help="pool candidate passages from every configuration")
    b.add_argument("--strategies", help="comma separated; default = every built strategy")
    b.add_argument("--depth", type=int, default=3, help="top-N per configuration to pool (default 3)")
    b.add_argument("--tol", type=float, default=30.0, help="same tolerance you evaluate with")
    b.add_argument("--include-rerank", action="store_true", help="also pool the (slow) reranker configurations")
    b.set_defaults(func=cmd_build)

    j = sub.add_parser("judge", parents=[common], help="y/n judging, shuffled and config-blind")
    j.add_argument("--seed", type=int, default=0)
    j.set_defaults(func=cmd_judge)

    s = sub.add_parser("status", parents=[common], help="judging progress")
    s.set_defaults(func=cmd_status)

    a = sub.add_parser("apply", parents=[common], help="write the audited question file")
    a.add_argument("--out", type=Path, default=EVAL_DIR / "questions.audited.json")
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
