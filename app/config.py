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
    # Per-user bot cookies (Netscape exports uploaded through /bot-users).
    cookies_root: str = str(BACKEND_DIR / "storage" / "cookies")
    admin_email: str = "admin@recsterio.local"
    admin_password: str = "change-me"
    access_token_ttl_minutes: int = 720
    log_level: str = "INFO"

    # ---- bot / service auth ----
    bot_api_token: str = "replace-me-with-a-shared-service-token"

    # ---- telegram account linking (console <-> bot) ----
    # Deep-link host for the link flow: the console embeds it as
    # https://t.me/<username>?start=link-<code>. Leave unset in dev (the
    # console then shows the bare code); set e.g. TELEGRAM_BOT_USERNAME=
    # recsteriobot in deploy so the code opens a chat with the bot directly.
    telegram_bot_username: Optional[str] = None
    # How long a console-generated link code stays redeemable by the bot.
    link_code_ttl_minutes: int = 15

    # ---- google sign-in (instruction.txt #2) ----
    # Leave the client id empty to disable the "Continue with Google" button; the
    # API then answers with a clear 400 instead of redirecting to a broken URL.
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None

    # ---- personal-cloud storage (contributor's own Google Drive / OneDrive) ----
    # A contributor may keep their video bytes on their own cloud instead of the
    # server disk. The OAuth apps are registered by the operator; these are their
    # credentials. Leave a provider's client id empty and that provider is not
    # offered anywhere.
    onedrive_client_id: Optional[str] = None
    onedrive_client_secret: Optional[str] = None
    onedrive_tenant: str = "common"
    # Where the portal (or console) is told to come back after the provider's
    # consent screen. Must be registered as a redirect URI in both OAuth apps.
    drive_oauth_redirect: str = "http://localhost:5174/settings"
    # Where share links send the recipient: the public dataset portal, which is
    # a separate app from the ops console so the console never has to be exposed.
    portal_frontend_url: str = "http://localhost:5174"
    # Guest tokens (public browsing, passcode and share redemption).
    guest_ttl_minutes: int = 1440

    # ---- gemini ----
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-pro"
    gemini_audio_chunk_seconds: int = 600

    # ---- local ASR ----
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # ---- downloader (yt-dlp) ----
    # TikTok and YouTube increasingly answer datacenter IPs with a bot check.
    # Supplying a cookies file (exported from a logged-in browser session the
    # operator is entitled to use) is the supported way to get past it. Leave
    # unset to browse anonymously. Browser cookies and TLS impersonation are
    # the local-dev equivalents: reuse the operator's own browser session and
    # send a real browser TLS fingerprint.
    ytdlp_cookies_file: Optional[str] = None
    ytdlp_proxy: Optional[str] = None
    ytdlp_user_agent: Optional[str] = None
    ytdlp_cookies_from_browser: Optional[str] = None  # "chrome" or "chrome:Profile 1" — local dev only
    ytdlp_impersonate: Optional[str] = None           # e.g. "chrome"; needs curl-cffi

    # Direct file upload (POST /videos/upload). Ingestion must not depend on a
    # platform's willingness to serve a download.
    max_upload_mb: int = 512
    # Largest accepted per-user cookies.txt upload (POST /bot-users/{id}/cookies).
    max_cookies_kb: int = 2048

    # ---- worker ----
    worker_poll_seconds: float = 3.0
    worker_max_attempts: int = 3
    worker_concurrency: int = 2
    worker_autostart: bool = True
    # Cross-process heartbeat file (read by GET /health when the worker runs as
    # a separate `python -m app.worker` process).
    worker_heartbeat_file: str = str(BACKEND_DIR / "storage" / "worker_heartbeat.json")

    # ---- http / security ----
    # Both frontends call the API through a same-origin proxy (Vite in dev, Caddy
    # in deploy), so this allowlist is a dev convenience for anyone who points
    # VITE_API_BASE_URL at an absolute URL — not a requirement of the normal setup.
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:5174"]
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
            return ["http://localhost:5173", "http://localhost:5174"]
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return list(v)

    @property
    def media_root_path(self) -> Path:
        p = Path(self.media_root).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    @property
    def cookies_root_path(self) -> Path:
        p = Path(self.cookies_root).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    def ensure_dirs(self) -> None:
        self.media_root_path.mkdir(parents=True, exist_ok=True)
        self.cookies_root_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
