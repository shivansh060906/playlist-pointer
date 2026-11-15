"""Step 1: download transcripts for every video in the playlist.

    python -m scripts.ingest --limit 5            # smoke test on the first 5 videos
    python -m scripts.ingest                      # whole playlist (resumable)
    python -m scripts.ingest --whisper-fallback   # transcribe locally when captions are missing
"""
from __future__ import annotations

import argparse
import json

from app.config import get_settings
from app.ingest.youtube import ingest_playlist, list_playlist


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playlist", default=settings.playlist_url, help="YouTube playlist URL")
    ap.add_argument("--limit", type=int, help="only the first N videos (for testing)")
    ap.add_argument("--list-only", action="store_true", help="print the playlist and exit")
    ap.add_argument("--whisper-fallback", action="store_true", help="use faster-whisper when captions fail")
    ap.add_argument("--force-whisper", action="store_true", help="always use faster-whisper (ignores captions)")
    ap.add_argument("--whisper-model", default="small.en", help="faster-whisper model size")
    ap.add_argument("--overwrite", action="store_true", help="re-fetch transcripts that already exist")
    args = ap.parse_args()

    print(f"Listing playlist: {args.playlist}")
    entries = list_playlist(args.playlist)
    if args.limit:
        entries = entries[: args.limit]
    print(f"Found {len(entries)} videos. First few:")
    for e in entries[:3]:
        print(f"  {e['playlist_index']:>3}. {e['title']}")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "playlist.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    if args.list_only:
        for e in entries:
            print(f"{e['playlist_index']:>3}. {e['video_id']}  {e['title']}")
        return

    stats = ingest_playlist(
        entries,
        settings.transcripts_dir,
        whisper_fallback=args.whisper_fallback,
        force_whisper=args.force_whisper,
        whisper_model=args.whisper_model,
        overwrite=args.overwrite,
    )
    print(f"\nSaved {stats['saved']}, skipped {stats['skipped']}, failed {len(stats['failed'])}")
    for vid, reason in stats["failed"].items():
        print(f"  FAILED {vid}: {reason}")
    if stats["failed"]:
        print("Re-run with --whisper-fallback to transcribe the failed videos locally.")


if __name__ == "__main__":
    main()
