"""BM25 tokenizer that understands C++.

A generic tokenizer turns `std::vector` into 'std' + 'vector' and `C++` into 'c', which throws away
the exact signal keyword search is supposed to provide. This one keeps `std::vector` as a token (and
also emits its parts), maps 'c++' / 'c plus plus' to 'cpp', and applies a tiny plural stemmer so
'pointers' matches 'pointer'. The same function is used for documents and queries, so stemming
quirks cancel out.
"""
from __future__ import annotations

import re

_PHRASES = [
    (re.compile(r"(?<![a-z0-9_])c\s*\+\+", re.IGNORECASE), " cpp "),
    (re.compile(r"\bc\s+plus\s+plus\b", re.IGNORECASE), " cpp "),
    (re.compile(r"(?<![a-z0-9_])c#", re.IGNORECASE), " csharp "),
]
_TOKEN = re.compile(r"[a-z_][a-z0-9_]*(?:::[a-z_][a-z0-9_]*)+|[a-z_][a-z0-9_]*|\d+")

STOPWORDS = frozenset(
    """a an the and or but if then so of to in on at by for with from as is are was were be been being
    it its this that these those i you we they he she them my your our their me us do does did done
    have has had will would can could should just also very really like kind sort bit actually basically
    okay ok right yeah um uh gonna wanna going get got what which who whom how why when where""".split()
)


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    for pattern, replacement in _PHRASES:
        lowered = pattern.sub(replacement, lowered)

    tokens: list[str] = []
    for match in _TOKEN.findall(lowered):
        if "::" in match:
            tokens.append(match)
            tokens.extend(part for part in match.split("::") if part not in STOPWORDS)
        elif match not in STOPWORDS:
            tokens.append(_stem(match))
    return tokens
