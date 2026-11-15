from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Segment:
    """One caption line: text plus where it sits in the video."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Transcript:
    video_id: str
    title: str
    playlist_index: int
    source: str  # youtube-auto | youtube-manual | whisper
    segments: list[Segment]

    @property
    def duration(self) -> float:
        return max((s.end for s in self.segments), default=0.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_id": self.video_id,
            "title": self.title,
            "playlist_index": self.playlist_index,
            "source": self.source,
            "segments": [asdict(s) for s in self.segments],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            video_id=raw["video_id"],
            title=raw["title"],
            playlist_index=raw.get("playlist_index", 0),
            source=raw.get("source", "unknown"),
            segments=[Segment(**s) for s in raw["segments"]],
        )


@dataclass
class Chunk:
    chunk_id: str
    video_id: str
    title: str
    playlist_index: int
    start: float
    end: float
    text: str
    strategy: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def embed_text(self) -> str:
        """What gets embedded / BM25-indexed / reranked. The title carries a lot of topic signal
        ('Pointers in C++', 'Templates in C++') that individual caption lines lack."""
        return f"{self.title}\n{self.text}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Chunk":
        return cls(**json.loads(line))


def load_all_transcripts(directory: Path) -> list[Transcript]:
    transcripts = [Transcript.load(p) for p in sorted(directory.glob("*.json"))]
    return sorted(transcripts, key=lambda t: (t.playlist_index, t.video_id))
