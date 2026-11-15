"""Step 4: run the ablation. Every chunking strategy x retrieval configuration over your labelled questions.

    python -m scripts.eval                          # uses eval/questions.json
    python -m scripts.eval --skip-rerank            # much faster: drops the slow cross-encoder rows
    python -m scripts.eval --tol 15 --strategies semantic,fixed
    python -m scripts.eval --misses-strategy sliding --misses-config "dense only"

Writes to eval/results/:
  results.md / results.json   table with 95% bootstrap CIs (the table you put in your README / CV)
  paired.json                 paired differences behind the "Configurations vs ..." tables in results.md
  per_question.json           rank + seek error of every question under every configuration
  misses.md / misses.json     every question that missed at hit@1, classified (label error / wrong video / ...)

Weighted-RRF rows ("hybrid RRF 2:1") give dense twice the weight of BM25 in the fusion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.config import BACKEND_DIR, STRATEGIES, get_settings
from app.engine import RAGEngine
from app.eval_runner import CONFIGS, evaluate_and_report
from app.evaluation import load_questions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=BACKEND_DIR / "eval" / "questions.json")
    ap.add_argument("--strategies", help="comma separated; default = every strategy that has been built")
    ap.add_argument("--tol", type=float, default=30.0, help="timestamp tolerance in seconds (default 30)")
    ap.add_argument("--depth", type=int, default=10, help="how many results to retrieve per query")
    ap.add_argument("--out", type=Path, default=BACKEND_DIR / "eval" / "results")
    ap.add_argument("--skip-rerank", action="store_true", help="skip the (slow) cross-encoder configurations")
    ap.add_argument("--baseline", default="dense only", help="configuration the paired tables compare against")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for the confidence intervals")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (results are reproducible)")
    ap.add_argument("--misses-strategy", default="semantic", help="chunking strategy for the misses report")
    ap.add_argument("--misses-config", default="hybrid + glossary", help="configuration for the misses report")
    ap.add_argument("--no-misses", action="store_true", help="do not write the misses report")
    args = ap.parse_args()

    configs = [c for c in CONFIGS if not (args.skip_rerank and c[1].get("rerank"))]
    labels = [label for label, _ in configs]
    if args.baseline not in labels:
        ap.error(f"--baseline must be one of: {labels}")

    if not args.questions.exists():
        raise SystemExit(
            f"{args.questions} not found. Copy eval/questions.example.json to eval/questions.json and "
            "fill it with your own labelled questions (see README, 'Building the eval set')."
        )
    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit("No usable questions found (placeholders starting with REPLACE are skipped).")

    engine = RAGEngine(get_settings())
    strategies = args.strategies.split(",") if args.strategies else engine.available_strategies()
    for s in strategies:
        if s not in STRATEGIES:
            raise SystemExit(f"unknown strategy {s!r}")
    if not strategies:
        raise SystemExit("No indexes built. Run: python -m scripts.build_index --strategy all")

    print(
        f"{len(questions)} questions | tolerance +/-{args.tol:g}s | strategies: {', '.join(strategies)} | "
        f"{len(configs)} configurations each\n"
    )
    evaluate_and_report(
        {s: engine.retriever(s) for s in strategies},
        questions,
        args.out,
        configs=configs,
        baseline=args.baseline,
        tol=args.tol,
        depth=args.depth,
        n_boot=args.bootstrap,
        seed=args.seed,
        misses_strategy=None if args.no_misses else args.misses_strategy,
        misses_config=args.misses_config,
    )


if __name__ == "__main__":
    main()
