"""API request/response models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .retrieval.hybrid import Hit, VideoGroup
from .utils import format_timestamp, youtube_url

Mode = Literal["hybrid", "dense", "bm25"]


class HitOut(BaseModel):
    chunk_id: str
    video_id: str
    title: str
    playlist_index: int
    start: float
    end: float
    seek_seconds: int  # where the player should jump (start minus a small lead-in)
    timestamp: str  # human readable, e.g. "12:34"
    url: str
    text: str
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


class VideoGroupOut(BaseModel):
    video_id: str
    title: str
    playlist_index: int
    best_score: float
    hit_count: int
    seek_seconds: int
    timestamp: str
    url: str


class SearchResponse(BaseModel):
    query: str
    strategy: str
    mode: Mode
    rerank_applied: bool
    expanded_query: str | None = None
    timings_ms: dict[str, float]
    hits: list[HitOut]
    videos: list[VideoGroupOut] | None = None
    out_of_scope: bool = False  # query looks unrelated to the lectures; hits are only the closest matches
    relevance: float | None = None  # best dense cosine similarity (debug/calibration)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(6, ge=1, le=12)
    strategy: str | None = None
    mode: Mode = "hybrid"
    rerank: bool = False
    expand: bool = False
    video_ids: list[str] | None = None
    playlist: str | None = None


class Citation(BaseModel):
    index: int  # the [n] used in the answer text
    hit: HitOut


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    citations: list[Citation]
    hits: list[HitOut]
    timings_ms: dict[str, float]
    out_of_scope: bool = False
    relevance: float | None = None


class PlaylistAddRequest(BaseModel):
    url: str = Field(min_length=10, max_length=300)


class PlaylistOut(BaseModel):
    id: str
    title: str
    url: str | None = None
    status: Literal["ready", "queued", "listing", "fetching", "indexing", "failed", "interrupted"]
    video_count: int = 0
    done: int = 0
    total: int = 0
    skipped: int = 0
    truncated_from: int | None = None
    error: str | None = None


class VideoOut(BaseModel):
    video_id: str
    title: str
    playlist_index: int
    duration: float
    chunk_count: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    default_strategy: str
    strategies_built: list[str]
    llm_configured: bool
    llm_provider: str


def hit_out(hit: Hit, lead_in: int = 2) -> HitOut:
    c = hit.chunk
    seek = max(0, int(c.start) - lead_in)
    return HitOut(
        chunk_id=c.chunk_id,
        video_id=c.video_id,
        title=c.title,
        playlist_index=c.playlist_index,
        start=c.start,
        end=c.end,
        seek_seconds=seek,
        timestamp=format_timestamp(seek),
        url=youtube_url(c.video_id, seek),
        text=c.text,
        score=round(hit.score, 4),
        dense_rank=hit.dense_rank,
        bm25_rank=hit.bm25_rank,
        rerank_score=None if hit.rerank_score is None else round(hit.rerank_score, 4),
    )


def video_group_out(group: VideoGroup, lead_in: int = 2) -> VideoGroupOut:
    seek = max(0, int(group.best_chunk.start) - lead_in)
    return VideoGroupOut(
        video_id=group.video_id,
        title=group.title,
        playlist_index=group.playlist_index,
        best_score=round(group.best_score, 4),
        hit_count=group.hit_count,
        seek_seconds=seek,
        timestamp=format_timestamp(seek),
        url=youtube_url(group.video_id, seek),
    )
