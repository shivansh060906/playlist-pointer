"""Playlist library: every playlist gets its own folder (transcripts + index), and users can add new
ones at runtime. Adding = list videos -> fetch captions -> chunk + embed + index (no eval step).

Layout:  <data_dir>/            the original/default playlist (back-compat with the CLI scripts)
         <data_dir>/playlists/<youtube list id>/{meta.json, transcripts/, index/, chroma/}
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import STRATEGIES, Settings
from .index.builder import build_index
from .index.store import index_exists
from .ingest import youtube
from .models import load_all_transcripts

DEFAULT_ID = "default"
DEFAULT_TITLE = "The Cherno's C++ series"
_LIST_PARAM = re.compile(r"[?&]list=([A-Za-z0-9_-]{10,})")
_BARE_ID = re.compile(r"^(PL|UU|OL|FL|LL|RD)[A-Za-z0-9_-]{8,}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
ACTIVE = ("queued", "listing", "fetching", "indexing")


class JobBusy(RuntimeError):
    pass


class UnknownPlaylist(ValueError):
    pass


def parse_playlist_id(text: str) -> str:
    """Accepts a playlist URL (…?list=PL…) or a bare id. A plain video link is rejected: we need a playlist."""
    text = text.strip()
    m = _LIST_PARAM.search(text)
    if m:
        return m.group(1)
    if _BARE_ID.match(text):
        return text
    raise ValueError("That doesn't look like a YouTube playlist link (expected …/playlist?list=PL…).")


@dataclass
class PlaylistInfo:
    id: str
    title: str
    url: str | None
    status: str  # ready | queued | listing | fetching | indexing | failed | interrupted
    video_count: int = 0
    done: int = 0  # videos processed so far (while fetching)
    total: int = 0
    error: str | None = None
    skipped: int = 0  # videos without usable captions
    truncated_from: int | None = None  # playlist had more videos than the cap


@dataclass
class _Job:
    status: str = "queued"
    done: int = 0
    total: int = 0
    error: str | None = None
    skipped: int = 0
    truncated_from: int | None = None
    title: str = ""
    url: str = ""


class Library:
    def __init__(
        self,
        settings: Settings,
        embedder_provider: Callable[[], object],
        *,
        on_change: Callable[[str], None] | None = None,
        delay: tuple[float, float] = (1.0, 2.5),
    ):
        self.settings = settings
        self._embedder = embedder_provider
        self._on_change = on_change or (lambda pid: None)
        self._delay = delay
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()
        self.root = settings.data_dir / "playlists"

    # ------------------------------------------------------------------ paths / lookup
    def settings_for(self, pid: str | None) -> Settings:
        pid = pid or DEFAULT_ID
        if pid == DEFAULT_ID:
            return self.settings
        if not _SAFE_ID.match(pid) or not (self.root / pid).is_dir():
            raise UnknownPlaylist(f"Unknown playlist {pid!r}")
        return dataclasses.replace(self.settings, data_dir=self.root / pid)

    def _meta_path(self, pid: str) -> Path:
        return self.root / pid / "meta.json"

    def _read_meta(self, pid: str) -> dict:
        try:
            return json.loads(self._meta_path(pid).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_meta(self, pid: str, **fields) -> None:
        meta = self._read_meta(pid) | fields
        path = self._meta_path(pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _is_built(self, settings: Settings) -> bool:
        return any(index_exists(settings, s) for s in STRATEGIES)

    def title(self, pid: str | None) -> str:
        pid = pid or DEFAULT_ID
        if pid == DEFAULT_ID:
            return DEFAULT_TITLE if self.settings.playlist_url else "this playlist"
        return self._read_meta(pid).get("title") or "this playlist"

    # ------------------------------------------------------------------ listing
    def info(self, pid: str) -> PlaylistInfo:
        with self._lock:
            job = self._jobs.get(pid)
            if job is not None:
                return PlaylistInfo(
                    pid, job.title or "Loading…", job.url, job.status, 0, job.done, job.total,
                    job.error, job.skipped, job.truncated_from,
                )
        if pid == DEFAULT_ID:
            return PlaylistInfo(pid, DEFAULT_TITLE, self.settings.playlist_url, "ready", self._video_count(self.settings))
        meta = self._read_meta(pid)
        st = self.settings_for(pid)
        status = "ready" if self._is_built(st) else ("failed" if meta.get("status") == "failed" else "interrupted")
        return PlaylistInfo(
            pid, meta.get("title") or pid, meta.get("url"), status, meta.get("video_count", 0),
            error=meta.get("error"), skipped=meta.get("skipped", 0), truncated_from=meta.get("truncated_from"),
        )

    @staticmethod
    def _video_count(settings: Settings) -> int:
        d = settings.transcripts_dir
        return len(list(d.glob("*.json"))) if d.exists() else 0

    def list(self) -> list[PlaylistInfo]:
        ids: list[str] = []
        if self._is_built(self.settings) or DEFAULT_ID in self._jobs:
            ids.append(DEFAULT_ID)
        if self.root.exists():
            ids += sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / "meta.json").exists())
        with self._lock:
            ids += [p for p in self._jobs if p not in ids]
        return [self.info(p) for p in ids]

    # ------------------------------------------------------------------ add / delete
    def add(self, url: str) -> PlaylistInfo:
        pid = parse_playlist_id(url)
        try:  # the originally configured playlist already lives at the default location
            if pid == parse_playlist_id(self.settings.playlist_url) and self._is_built(self.settings):
                return self.info(DEFAULT_ID)
        except ValueError:
            pass
        with self._lock:
            if any(j.status in ACTIVE for j in self._jobs.values()):
                raise JobBusy("Another playlist is still being indexed. Wait for it to finish.")
            if (self.root / pid).is_dir() and self._is_built(self.settings_for(pid)):
                return self.info(pid)
            self._jobs[pid] = _Job(url=url.strip())
        threading.Thread(target=self._run, args=(pid, url.strip()), daemon=True).start()
        return self.info(pid)

    def delete(self, pid: str) -> None:
        if pid == DEFAULT_ID:
            raise ValueError("The default playlist can't be removed from the UI.")
        with self._lock:
            job = self._jobs.get(pid)
            if job and job.status in ACTIVE:
                raise JobBusy("That playlist is still being indexed.")
            self._jobs.pop(pid, None)
        self.settings_for(pid)  # raises UnknownPlaylist
        self._on_change(pid)  # drop cached retrievers before the files go away
        shutil.rmtree(self.root / pid, ignore_errors=True)

    # ------------------------------------------------------------------ the job
    def _set(self, pid: str, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._jobs[pid], k, v)

    def _run(self, pid: str, url: str) -> None:
        try:
            self._set(pid, status="listing")
            title, entries = youtube.list_playlist_info(url)
            if not entries:
                raise RuntimeError("That playlist has no available videos (is it private?).")
            cap = self.settings.max_playlist_videos
            truncated = len(entries) if len(entries) > cap else None
            entries = entries[:cap]
            self._set(pid, title=title, total=len(entries), truncated_from=truncated)
            st = self.settings_for_new(pid)
            self._write_meta(pid, id=pid, url=url, title=title, status="fetching", created=time.time())

            self._set(pid, status="fetching")
            stats = youtube.ingest_playlist(
                entries, st.transcripts_dir, delay=self._delay, log=lambda *_: None,
                progress=lambda done, total: self._set(pid, done=done, total=total),
            )
            saved = len(load_all_transcripts(st.transcripts_dir))
            if saved == 0:
                reasons = sorted(set(stats["failed"].values()))[:1]
                raise RuntimeError(
                    "No captions could be fetched for any video"
                    + (f" ({reasons[0]})." if reasons else ".")
                    + " Videos need English captions (manual or auto-generated)."
                )
            skipped = len(entries) - saved
            self._set(pid, status="indexing", skipped=skipped)
            self._write_meta(pid, status="indexing")
            build_index(st, st.default_strategy, self._embedder(), log=lambda *_: None)

            self._write_meta(pid, status="ready", video_count=saved, skipped=skipped, truncated_from=truncated, error=None)
            self._on_change(pid)
            with self._lock:
                self._jobs.pop(pid, None)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc) or type(exc).__name__
            if (self.root / pid).is_dir():
                self._write_meta(pid, status="failed", error=msg)
            self._set(pid, status="failed", error=msg)

    def settings_for_new(self, pid: str) -> Settings:
        (self.root / pid).mkdir(parents=True, exist_ok=True)
        return self.settings_for(pid)
