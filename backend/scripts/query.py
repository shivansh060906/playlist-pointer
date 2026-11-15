"""Quick CLI to try the retriever / full RAG answer without starting the server.

    python -m scripts.query "why should I use references instead of pointers"
    python -m scripts.query "what is nullptr" --mode bm25 --expand
    python -m scripts.query "explain move semantics" --ask
    python -m scripts.query "how do virtual functions work" --rerank

Defaults match the evaluated configuration: hybrid, weighted RRF (RRF_WEIGHTS), no rerank, no glossary.
"""
from __future__ import annotations

import argparse
import textwrap

from app.config import STRATEGIES, get_settings
from app.engine import RAGEngine
from app.retrieval.hybrid import MODES
from app.schemas import hit_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--strategy", choices=STRATEGIES)
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("--rerank", action="store_true", help="cross-encoder rerank (slow; hurt accuracy in our eval)")
    ap.add_argument("--expand", action="store_true", help="glossary query expansion for BM25 (no measured gain)")
    ap.add_argument("--ask", action="store_true", help="also generate a cited answer with the LLM")
    args = ap.parse_args()

    settings = get_settings()
    engine = RAGEngine(settings)
    lead = settings.lead_in_seconds
    kwargs = dict(strategy=args.strategy, mode=args.mode, rerank=args.rerank, expand=args.expand)

    if args.ask:
        result = engine.ask(args.query, max(args.k, 4), **kwargs)
        hits, timings = result.hits, result.timings_ms
        print(f"\n{result.answer}\n")
    else:
        result = engine.search(args.query, args.k, **kwargs)
        hits, timings = result.hits, result.timings_ms
        if result.expanded_query:
            print(f"(BM25 query expanded to: {result.expanded_query})")

    for i, hit in enumerate(hits, start=1):
        out = hit_out(hit, lead)
        print(f"{i}. #{out.playlist_index} {out.title}  @ {out.timestamp}  (score {out.score})")
        print(f"   {out.url}")
        print(textwrap.indent(textwrap.shorten(out.text, 220), "   "))
    print(f"\ntimings (ms): {timings}")


if __name__ == "__main__":
    main()
