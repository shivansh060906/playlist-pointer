"""Three chunking strategies over timestamped caption segments.

Every chunk keeps real start/end times, which is what makes timestamp-accurate citations possible.
All strategies snap to caption-segment boundaries so a chunk never starts mid-caption.

  fixed     non-overlapping windows of ~N words (baseline)
  sliding   overlapping windows (stride < window) so an idea cut by a boundary appears whole somewhere
  semantic  split where adjacent blocks of text stop being about the same thing (embedding similarity dips)
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

from .models import Chunk, Segment, Transcript

EmbedFn = Callable[[Sequence[str]], np.ndarray]

# Only strip well-known caption tags and speaker-change markers, never arbitrary [brackets].
_NOISE = re.compile(
    r"\[(?:music|applause|laughter|noise|silence|inaudible|__)[^\]]*\]|>>+|♪+", re.IGNORECASE
)
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS.sub(" ", _NOISE.sub(" ", text)).strip()


def clean_segments(segments: Sequence[Segment]) -> list[Segment]:
    cleaned = []
    for seg in segments:
        text = clean_text(seg.text)
        if text:
            cleaned.append(Segment(text=text, start=seg.start, duration=seg.duration))
    return cleaned


def _words(text: str) -> int:
    return len(text.split())


def _make_chunk(t: Transcript, segs: Sequence[Segment], strategy: str, n: int) -> Chunk:
    return Chunk(
        chunk_id=f"{t.video_id}:{strategy}:{n:04d}",
        video_id=t.video_id,
        title=t.title,
        playlist_index=t.playlist_index,
        start=round(segs[0].start, 2),
        end=round(max(s.end for s in segs), 2),
        text=" ".join(s.text for s in segs),
        strategy=strategy,
    )


# --------------------------------------------------------------------------- fixed
def fixed_chunks(t: Transcript, target_words: int = 200, min_tail_words: int = 60) -> list[Chunk]:
    segs = clean_segments(t.segments)
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    words = 0
    for seg in segs:
        current.append(seg)
        words += _words(seg.text)
        if words >= target_words:
            groups.append(current)
            current, words = [], 0
    if current:
        if groups and words < min_tail_words:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return [_make_chunk(t, g, "fixed", i) for i, g in enumerate(groups)]


# ------------------------------------------------------------------------- sliding
def sliding_chunks(t: Transcript, window_words: int = 200, stride_words: int = 100) -> list[Chunk]:
    segs = clean_segments(t.segments)
    if not segs:
        return []
    n = len(segs)
    cum = [0]  # cum[i] = number of words before segment i
    for seg in segs:
        cum.append(cum[-1] + _words(seg.text))

    chunks: list[Chunk] = []
    i = 0
    while i < n:
        j = i
        while j < n and cum[j + 1] - cum[i] < window_words:
            j += 1
        j = min(j, n - 1)
        chunks.append(_make_chunk(t, segs[i : j + 1], "sliding", len(chunks)))
        if j >= n - 1:
            break
        target = cum[i] + stride_words
        nxt = i + 1
        while nxt < n and cum[nxt] < target:
            nxt += 1
        i = max(nxt, i + 1)
    return chunks


# ------------------------------------------------------------------------ semantic
def _blocks(segs: Sequence[Segment], block_words: int) -> list[list[Segment]]:
    blocks: list[list[Segment]] = []
    current: list[Segment] = []
    words = 0
    for seg in segs:
        current.append(seg)
        words += _words(seg.text)
        if words >= block_words:
            blocks.append(current)
            current, words = [], 0
    if current:
        if blocks and words < block_words // 2:
            blocks[-1].extend(current)
        else:
            blocks.append(current)
    return blocks


def semantic_chunks(
    t: Transcript,
    embed_fn: EmbedFn,
    block_words: int = 30,
    min_words: int = 120,
    max_words: int = 320,
    percentile: float = 25.0,
    window: int = 2,
) -> list[Chunk]:
    """Topic-shift chunking (TextTiling-style, embedding based).

    1. Group captions into small blocks (~block_words words) -- single caption lines are too short to embed.
    2. Embed blocks; for each gap compare the mean embedding of the `window` blocks before vs after.
    3. A gap is a boundary candidate when its similarity is in the lowest `percentile` of this video
       and a local minimum. Chunks are kept within [min_words, max_words]; if max is hit without a
       good valley, cut at the deepest valley seen since min_words was reached.
    """
    segs = clean_segments(t.segments)
    if not segs:
        return []
    blocks = _blocks(segs, block_words)
    n = len(blocks)
    if n < 3:
        return [_make_chunk(t, segs, "semantic", 0)]

    texts = [" ".join(s.text for s in b) for b in blocks]
    embs = np.asarray(embed_fn(texts), dtype=np.float32)
    embs = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9, None)

    sims = np.empty(n - 1, dtype=np.float32)
    for i in range(n - 1):
        left = embs[max(0, i - window + 1) : i + 1].mean(axis=0)
        right = embs[i + 1 : i + 1 + window].mean(axis=0)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9
        sims[i] = float(left @ right) / denom
    threshold = float(np.percentile(sims, percentile))
    block_word_counts = [sum(_words(s.text) for s in b) for b in blocks]

    def is_local_min(i: int) -> bool:
        return (i == 0 or sims[i] <= sims[i - 1]) and (i == n - 2 or sims[i] <= sims[i + 1])

    groups: list[list[Segment]] = []
    start = 0
    while start < n:
        words = 0
        best, best_sim = None, float("inf")
        cut = n - 1
        for i in range(start, n):
            words += block_word_counts[i]
            if i == n - 1:
                cut = i
                break
            if words >= min_words:
                if sims[i] < best_sim:
                    best, best_sim = i, float(sims[i])
                if sims[i] <= threshold and is_local_min(i):
                    cut = i
                    break
            if words >= max_words:
                cut = best if best is not None else i
                break
        groups.append([s for b in blocks[start : cut + 1] for s in b])
        start = cut + 1

    if len(groups) > 1 and sum(_words(s.text) for s in groups[-1]) < min_words // 2:
        groups[-2].extend(groups.pop())

    return [_make_chunk(t, g, "semantic", i) for i, g in enumerate(groups)]


# ----------------------------------------------------------------------- dispatcher
def chunk_transcript(
    t: Transcript, strategy: str, embed_fn: EmbedFn | None = None, **kwargs
) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_chunks(t, **kwargs)
    if strategy == "sliding":
        return sliding_chunks(t, **kwargs)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs an embed_fn")
        return semantic_chunks(t, embed_fn, **kwargs)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
