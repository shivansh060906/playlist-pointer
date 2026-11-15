"""Query expansion from a C++ glossary.

Auto-captions mangle jargon ('cout' -> 'see out', 'lvalue' -> 'l value', 'nullptr' -> 'null pointer').
Whisper transcripts are better but not perfect either. Rather than rewrite transcripts, we expand the
BM25 query with known variants, so 'what is nullptr' also searches for 'null pointer'.
Dense retrieval is unaffected; this specifically repairs the keyword side.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "cpp_glossary.json"


class Glossary:
    def __init__(self, entries: dict[str, list[str]]):
        self._groups: list[tuple[list[str], list[re.Pattern[str]]]] = []
        for canonical, variants in entries.items():
            terms = [canonical.lower()] + [v.lower() for v in variants]
            patterns = [re.compile(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])") for term in terms]
            self._groups.append((terms, patterns))

    @classmethod
    def load(cls, path: Path | None = None) -> "Glossary":
        raw = json.loads((path or _DEFAULT_PATH).read_text(encoding="utf-8"))
        return cls({k: v for k, v in raw.items() if not k.startswith("_")})

    def expand(self, query: str) -> str:
        """Return the query plus variants of every glossary entry it mentions."""
        lowered = query.lower()
        additions: list[str] = []
        for terms, patterns in self._groups:
            if any(p.search(lowered) for p in patterns):
                for term in terms:
                    if term not in lowered and term not in additions:
                        additions.append(term)
        return query if not additions else f"{query} {' '.join(additions)}"
