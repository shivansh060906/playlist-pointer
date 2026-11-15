"""Helper for writing the eval set: find *where* things are said in your downloaded transcripts.

    python -m scripts.peek "null pointer"                 # every place the phrase occurs (timestamped)
    python -m scripts.peek --video VIDEO_ID --at 12:30    # read the transcript around a moment
    python -m scripts.peek --videos                       # list video ids + titles
"""
from __future__ import annotations

import argparse
import re

from app.chunking import clean_text
from app.config import get_settings
from app.models import load_all_transcripts
from app.utils import format_timestamp, parse_timestamp


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phrase", nargs="?", help="case-insensitive phrase to search for")
    ap.add_argument("--video", help="video id (use with --at)")
    ap.add_argument("--at", help="timestamp, e.g. 754 or 12:34")
    ap.add_argument("--window", type=float, default=45, help="seconds of context either side (with --at)")
    ap.add_argument("--videos", action="store_true", help="list available videos")
    ap.add_argument("--max", type=int, default=40, help="max matches to print")
    args = ap.parse_args()

    transcripts = load_all_transcripts(settings.transcripts_dir)
    if not transcripts:
        raise SystemExit(f"No transcripts in {settings.transcripts_dir}. Run: python -m scripts.ingest")

    if args.videos:
        for t in transcripts:
            print(f"{t.playlist_index:>3}  {t.video_id}  {t.title}  [{t.source}, {format_timestamp(t.duration)}]")
        return

    if args.video and args.at:
        center = parse_timestamp(args.at)
        t = next((t for t in transcripts if t.video_id == args.video), None)
        if t is None:
            raise SystemExit(f"Unknown video id {args.video!r}. Try --videos")
        print(f"{t.title}  ({t.video_id})")
        for seg in t.segments:
            if center - args.window <= seg.start <= center + args.window:
                marker = ">>" if abs(seg.start - center) < 3 else "  "
                print(f"{marker} {format_timestamp(seg.start):>7}  {clean_text(seg.text)}")
        return

    if not args.phrase:
        ap.error("give a phrase to search for, or use --videos / --video + --at")

    pattern = re.compile(re.escape(args.phrase), re.IGNORECASE)
    shown = 0
    for t in transcripts:
        segs = t.segments
        for i, seg in enumerate(segs):
            # match across the seam of two adjacent lines, since captions split phrases arbitrarily
            window = clean_text(seg.text + " " + (segs[i + 1].text if i + 1 < len(segs) else ""))
            if pattern.search(window):
                print(f"#{t.playlist_index:<3} {t.video_id}  {format_timestamp(seg.start):>7}  {t.title}")
                print(f"      ...{window}...")
                shown += 1
                if shown >= args.max:
                    print(f"(stopped after {args.max}; use --max to see more)")
                    return
    if shown == 0:
        print("No matches. (Auto-captions often mangle jargon -- try a simpler or alternate spelling.)")


if __name__ == "__main__":
    main()
