"""FastAPI server.  Run from backend/:  uvicorn app.main:app --reload --port 8000"""
from __future__ import annotations

import dataclasses
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .engine import RAGEngine
from .index.store import IndexNotBuilt
from .library import JobBusy, UnknownPlaylist
from .llm import LLMNotConfigured
from .retrieval.hybrid import group_hits_by_video
from .schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    Mode,
    PlaylistAddRequest,
    PlaylistOut,
    SearchResponse,
    VideoOut,
    hit_out,
    video_group_out,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = RAGEngine(settings)
    app.state.engine = engine
    if settings.warmup:  # load models in the background so startup is instant
        threading.Thread(target=engine.warmup, daemon=True).start()
    yield


app = FastAPI(title="Cherno C++ Lecture Search", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine(request: Request) -> RAGEngine:
    return request.app.state.engine


@app.exception_handler(IndexNotBuilt)
async def _index_not_built(_: Request, exc: IndexNotBuilt):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMNotConfigured)
async def _llm_not_configured(_: Request, exc: LLMNotConfigured):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/api/health", response_model=HealthResponse)
def health(engine: RAGEngine = Depends(get_engine)):
    return HealthResponse(
        default_strategy=engine.settings.default_strategy,
        strategies_built=engine.available_strategies(),
        llm_configured=engine.llm_configured(),
        llm_provider=engine.settings.llm_provider,
    )


@app.get("/api/videos", response_model=list[VideoOut])
def videos(playlist: str | None = None, engine: RAGEngine = Depends(get_engine)):
    try:
        return engine.videos(playlist)
    except UnknownPlaylist as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/playlists", response_model=list[PlaylistOut])
def playlists(engine: RAGEngine = Depends(get_engine)):
    return [dataclasses.asdict(p) for p in engine.library.list()]


@app.post("/api/playlists", response_model=PlaylistOut, status_code=202)
def add_playlist(body: PlaylistAddRequest, engine: RAGEngine = Depends(get_engine)):
    """Start indexing a playlist in the background; poll GET /api/playlists/{id} for progress."""
    try:
        return dataclasses.asdict(engine.library.add(body.url))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except JobBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/playlists/{playlist_id}", response_model=PlaylistOut)
def playlist_status(playlist_id: str, engine: RAGEngine = Depends(get_engine)):
    try:
        engine.library.settings_for(playlist_id)
    except UnknownPlaylist:
        if playlist_id not in {p.id for p in engine.library.list()}:  # a job may exist before its folder
            raise HTTPException(status_code=404, detail=f"Unknown playlist {playlist_id!r}") from None
    return dataclasses.asdict(engine.library.info(playlist_id))


@app.delete("/api/playlists/{playlist_id}", status_code=204)
def delete_playlist(playlist_id: str, engine: RAGEngine = Depends(get_engine)):
    try:
        engine.library.delete(playlist_id)
    except UnknownPlaylist as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2, max_length=300, description="Search query"),
    k: int = Query(5, ge=1, le=20),
    strategy: str | None = Query(None, description="fixed | sliding | semantic"),
    mode: Mode = "hybrid",
    rerank: bool = False,
    expand: bool = False,
    video_id: list[str] | None = Query(None, description="Restrict to these videos (repeatable)"),
    group_by: Literal["none", "video"] = "none",
    playlist: str | None = Query(None, description="Playlist id (default: the built-in playlist)"),
    engine: RAGEngine = Depends(get_engine),
):
    # Grouping by video needs a deeper pool, otherwise one chatty video fills all k slots.
    fetch_k = max(k, 20) if group_by == "video" else k
    try:
        result = engine.search(
            q, fetch_k, strategy=strategy, mode=mode, rerank=rerank, expand=expand, video_ids=video_id,
            playlist=playlist,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    lead = engine.settings.lead_in_seconds
    groups = None
    if group_by == "video":
        groups = [video_group_out(g, lead) for g in group_hits_by_video(result.hits)[:k]]
    return SearchResponse(
        query=q,
        strategy=strategy or engine.settings.default_strategy,
        mode=result.mode,  # type: ignore[arg-type]
        rerank_applied=result.rerank_applied,
        expanded_query=result.expanded_query,
        timings_ms=result.timings_ms,
        hits=[hit_out(h, lead) for h in result.hits],
        videos=groups,
        out_of_scope=result.out_of_scope,
        relevance=None if result.top_dense is None else round(result.top_dense, 4),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest, engine: RAGEngine = Depends(get_engine)):
    try:
        result = engine.ask(
            body.question,
            body.k,
            strategy=body.strategy,
            mode=body.mode,
            rerank=body.rerank,
            expand=body.expand,
            video_ids=body.video_ids,
            playlist=body.playlist,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IndexNotBuilt:
        raise
    except LLMNotConfigured:
        raise
    except Exception as exc:  # noqa: BLE001  (upstream LLM failure)
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    lead = engine.settings.lead_in_seconds
    hits = [hit_out(h, lead) for h in result.hits]
    return AskResponse(
        question=body.question,
        answer=result.answer,
        model=result.model,
        citations=[Citation(index=i, hit=hits[i - 1]) for i in result.cited],
        hits=hits,
        timings_ms=result.timings_ms,
        out_of_scope=result.out_of_scope,
        relevance=None if result.top_dense is None else round(result.top_dense, 4),
    )
