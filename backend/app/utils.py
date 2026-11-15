from __future__ import annotations

import re

_TS_RE = re.compile(r"^\d+(?::\d{1,2}){0,2}(?:\.\d+)?$")


def format_timestamp(seconds: float) -> str:
    """92 -> '1:32', 3725 -> '1:02:05'."""
    total = int(max(0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def parse_timestamp(value: str | int | float) -> float:
    """Accepts 754, '754', '12:34' or '1:02:05' and returns seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not _TS_RE.match(text):
        raise ValueError(f"Bad timestamp: {value!r}")
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def youtube_url(video_id: str, seconds: float = 0) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"
