"""Playlist listing + transcript fetching (YouTube captions, with a local Whisper fallback)."""
from __future__ import annotations

import random
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

from ..models import Segment, Transcript

# Biasing Whisper towards C++ vocabulary reduces ASR errors on exactly the terms people search for.
WHISPER_PROMPT = (
    "A C++ programming lesson by The Cherno. Terms: C++, std::vector, std::string, nullptr, const, "
    "constexpr, static, virtual, vtable, templates, lvalue, rvalue, move semantics, unique_ptr, "
    "shared_ptr, pointers, references, stack, heap, linker, compiler, preprocessor, header files, "
    "Visual Studio, CMake, premake, cout, iostream."
)

_UNAVAILABLE_TITLES = {"[Private video]", "[Deleted video]"}


def list_playlist(url: str) -> list[dict]:
    """Return [{video_id, title, playlist_index, duration}] in playlist order (no downloads)."""
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries: list[dict] = []
    for index, entry in enumerate(info.get("entries") or [], start=1):
        if not entry or not entry.get("id") or entry.get("title") in _UNAVAILABLE_TITLES:
            continue
        entries.append(
            {
                "video_id": entry["id"],
                "title": entry.get("title") or entry["id"],
                "playlist_index": index,
                "duration": entry.get("duration"),
            }
        )
    return entries


def fetch_captions(
    video_id: str, languages: tuple[str, ...] = ("en", "en-US", "en-GB")
) -> tuple[list[Segment], str]:
    """Fetch YouTube captions. Works with youtube-transcript-api both before and after v1.0."""
    from youtube_transcript_api import YouTubeTranscriptApi

    if hasattr(YouTubeTranscriptApi, "get_transcript"):  # <1.0
        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
        source = "youtube"
    else:  # >=1.0
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
        raw = fetched.to_raw_data()
        source = "youtube-auto" if getattr(fetched, "is_generated", True) else "youtube-manual"

    segments = [
        Segment(text=r["text"], start=float(r["start"]), duration=float(r.get("duration", 0.0)))
        for r in raw
        if r.get("text", "").strip()
    ]
    if not segments:
        raise ValueError("captions were empty")
    return segments, source


@lru_cache(maxsize=2)
def _whisper_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_with_whisper(
    video_id: str, model_size: str = "small.en", device: str = "auto", compute_type: str = "int8"
) -> list[Segment]:
    """Download the audio and transcribe locally. Needs `pip install -r requirements-whisper.txt`."""
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmp) / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        audio = next(Path(tmp).glob(f"{video_id}.*"))

        model = _whisper_model(model_size, device, compute_type)
        raw_segments, _info = model.transcribe(
            str(audio),
            language="en",
            vad_filter=True,
            initial_prompt=WHISPER_PROMPT,
            condition_on_previous_text=False,
        )
        return [
            Segment(text=s.text.strip(), start=float(s.start), duration=float(s.end - s.start))
            for s in raw_segments
            if s.text.strip()
        ]


def ingest_playlist(
    entries: list[dict],
    out_dir: Path,
    *,
    whisper_fallback: bool = False,
    force_whisper: bool = False,
    whisper_model: str = "small.en",
    overwrite: bool = False,
    delay: tuple[float, float] = (1.0, 2.5),
    log: Callable[[str], None] = print,
) -> dict:
    """Fetch and save one JSON transcript per video. Resumable: existing files are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"saved": 0, "skipped": 0, "failed": {}}  # type: dict

    for entry in entries:
        video_id = entry["video_id"]
        path = out_dir / f"{video_id}.json"
        if path.exists() and not overwrite:
            stats["skipped"] += 1
            continue

        used_whisper = False
        try:
            if force_whisper:
                segments, source = transcribe_with_whisper(video_id, whisper_model), "whisper"
                used_whisper = True
            else:
                try:
                    segments, source = fetch_captions(video_id)
                except Exception as exc:  # noqa: BLE001
                    if "Blocked" in type(exc).__name__:
                        raise RuntimeError(
                            "YouTube is rate-limiting/blocking this IP. Wait a while, switch network, "
                            "or use --force-whisper to bypass captions entirely."
                        ) from exc
                    if not whisper_fallback:
                        raise
                    log(f"  captions unavailable for {video_id} ({type(exc).__name__}); using Whisper")
                    segments, source = transcribe_with_whisper(video_id, whisper_model), "whisper"
                    used_whisper = True

            Transcript(
                video_id=video_id,
                title=entry["title"],
                playlist_index=entry["playlist_index"],
                source=source,
                segments=segments,
            ).save(path)
            stats["saved"] += 1
            log(f"[{entry['playlist_index']:>3}] {entry['title']}  ({source}, {len(segments)} segments)")
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            stats["failed"][video_id] = f"{type(exc).__name__}: {exc}"
            log(f"[{entry['playlist_index']:>3}] FAILED {entry['title']}: {type(exc).__name__}")

        if not used_whisper:
            time.sleep(random.uniform(*delay))

    return stats
