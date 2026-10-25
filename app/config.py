"""Application settings. All secrets come from the environment only."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(BACKEND_DIR / ".env"), str(BACKEND_DIR.parent / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- core ----
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/recsterio"
    secret_key: str = "change-me-to-a-long-random-string"
    media_root: str = str(BACKEND_DIR / "storage" / "media")
    admin_email: str = "admin@recsterio.local"
    admin_password: str = "change-me"
    access_token_ttl_minutes: int = 720
    log_level: str = "INFO"

    # ---- bot / service auth ----
    bot_api_token: str = "replace-me-with-a-shared-service-token"

    # ---- gemini ----
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-pro"
    gemini_audio_chunk_seconds: int = 600

    # ---- local ASR ----
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # ---- worker ----
    worker_poll_seconds: float = 3.0
    worker_max_attempts: int = 3
    worker_concurrency: int = 2
    worker_autostart: bool = True
    # Cross-process heartbeat file (read by GET /health when the worker runs as
    # a separate `python -m app.worker` process).
    worker_heartbeat_file: str = str(BACKEND_DIR / "storage" / "worker_heartbeat.json")

    # ---- http / security ----
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )
    video_create_rate_limit: int = 20  # per window
    video_create_rate_window: int = 60  # seconds
    login_rate_limit: int = 10
    login_rate_window: int = 60
    ffmpeg_timeout_seconds: int = 600

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if v is None or v == "":
            return ["http://localhost:5173"]
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return list(v)

    @property
    def media_root_path(self) -> Path:
        p = Path(self.media_root).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    def ensure_dirs(self) -> None:
        self.media_root_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
