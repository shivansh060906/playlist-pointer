"""Central configuration. Everything is overridable via environment variables or backend/.env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

STRATEGIES = ("fixed", "sliding", "semantic")

# The Cherno's main "C++" playlist. Override with PLAYLIST_URL or `ingest --playlist`.
# After ingesting, the first title should be "Welcome to C++".
DEFAULT_PLAYLIST = "https://www.youtube.com/playlist?list=PLlrATfBNZ98dudnM48yfGUldqGD0S4FFb"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    playlist_url: str
    embed_model: str
    rerank_model: str
    default_strategy: str
    llm_provider: str
    groq_api_key: str | None
    groq_model: str
    gemini_api_key: str | None
    gemini_model: str
    cors_origins: tuple[str, ...]
    lead_in_seconds: int
    warmup: bool
    rrf_weights: tuple[float, float]  # (dense, bm25) weights for hybrid fusion

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(_env("DATA_DIR", "data"))  # type: ignore[arg-type]
        if not data_dir.is_absolute():
            data_dir = BACKEND_DIR / data_dir

        strategy = _env("DEFAULT_STRATEGY", "semantic")
        if strategy not in STRATEGIES:
            raise ValueError(f"DEFAULT_STRATEGY must be one of {STRATEGIES}, got {strategy!r}")

        origins = _env("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")

        raw_weights = _env("RRF_WEIGHTS", "2,1") or "2,1"
        try:
            parsed = tuple(float(x) for x in raw_weights.split(","))
        except ValueError:
            parsed = ()
        if len(parsed) != 2 or min(parsed) < 0 or sum(parsed) <= 0:
            raise ValueError(f"RRF_WEIGHTS must look like '2,1' (dense,bm25), got {raw_weights!r}")

        return cls(
            data_dir=data_dir,
            playlist_url=_env("PLAYLIST_URL", DEFAULT_PLAYLIST),  # type: ignore[arg-type]
            embed_model=_env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),  # type: ignore[arg-type]
            rerank_model=_env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),  # type: ignore[arg-type]
            default_strategy=strategy,  # type: ignore[arg-type]
            llm_provider=(_env("LLM_PROVIDER", "groq") or "groq").lower(),
            groq_api_key=_env("GROQ_API_KEY"),
            groq_model=_env("GROQ_MODEL", "llama-3.3-70b-versatile"),  # type: ignore[arg-type]
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash"),  # type: ignore[arg-type]
            cors_origins=tuple(o.strip() for o in (origins or "").split(",") if o.strip()),
            # Captions lag the speech slightly, so links start a couple of seconds early.
            lead_in_seconds=int(_env("LEAD_IN_SECONDS", "2")),  # type: ignore[arg-type]
            warmup=(_env("WARMUP", "1") or "1").lower() not in ("0", "false", "no"),
            rrf_weights=(parsed[0], parsed[1]),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
