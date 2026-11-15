"""Step 2: chunk transcripts and build the vector + BM25 indexes.

    python -m scripts.build_index --strategy all
    python -m scripts.build_index --strategy semantic --min-words 100 --max-words 280
"""
from __future__ import annotations

import argparse
import json

from app.config import STRATEGIES, get_settings
from app.index.builder import build_index
from app.retrieval.embedder import SentenceTransformerEmbedder


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="all", choices=[*STRATEGIES, "all"])
    ap.add_argument("--target-words", type=int, help="fixed: words per chunk (default 200)")
    ap.add_argument("--window-words", type=int, help="sliding: window size (default 200)")
    ap.add_argument("--stride-words", type=int, help="sliding: stride (default 100)")
    ap.add_argument("--min-words", type=int, help="semantic: minimum chunk size (default 120)")
    ap.add_argument("--max-words", type=int, help="semantic: maximum chunk size (default 320)")
    ap.add_argument("--percentile", type=float, help="semantic: similarity percentile for a boundary (default 25)")
    args = ap.parse_args()

    options = {
        "fixed": {"target_words": args.target_words},
        "sliding": {"window_words": args.window_words, "stride_words": args.stride_words},
        "semantic": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "percentile": args.percentile,
        },
    }
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    embedder = SentenceTransformerEmbedder(settings.embed_model)
    all_stats = []
    for strategy in strategies:
        kwargs = {k: v for k, v in options[strategy].items() if v is not None}
        all_stats.append(build_index(settings, strategy, embedder, chunker_kwargs=kwargs))

    print("\nSummary")
    print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
