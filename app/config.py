"""Application settings. All secrets come from the environment only."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """The deployment would run with placeholder bootstrap secrets.

    Raised at settings-load time so `python -m app.worker`, `uvicorn` and the
    test bootstrap all hit the same wall: a default SECRET_KEY signs JWTs
    anyone with this repository can forge, a default ADMIN_PASSWORD is a
    public credential, and a default BOT_API_TOKEN hands the whole
    service-auth surface to whoever found it on GitHub.
    """


# Values a fresh checkout yields when the operator never set the env var.
# These are the strings the boot guard refuses in production.
PLACEHOLDER_DEFAULTS: dict[str, tuple[str, str]] = {
    #                env name      (field,             placeholder)
    "SECRET_KEY": ("secret_key", "change-me-to-a-long-random-string"),
    "ADMIN_PASSWORD": ("admin_password", "change-me"),
    "BOT_API_TOKEN": ("bot_api_token", "replace-me-with-a-shared-service-token"),
}


def _insecure_allowed() -> bool:
    """The explicit escape hatch for local development and the test suite."""
    return (os.environ.get("ALLOW_INSECURE") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def validate_security_settings(s: Settings) -> None:
    """Refuse placeholders for SECRET_KEY / ADMIN_PASSWORD / BOT_API_TOKEN.

    ALLOW_INSECURE=1 bypasses the guard for local dev and tests; anything else
    (including the var being absent) enforces it. Unset (`""`) counts as
    insecure too — the placeholder check above names the defaults, and an
    empty value disables the mechanism without ever being intended.
    """
    if _insecure_allowed():
        return
    offenders: list[str] = []
    for env_name, (field_name, placeholder) in PLACEHOLDER_DEFAULTS.items():
        value = (getattr(s, field_name) or "").strip()
        if not value or value == placeholder:
            offenders.append(env_name)
    if offenders:
        listed = "; ".join(offenders)
        raise ConfigError(
            "Refusing to start with insecure bootstrap settings: "
            f"{listed} {'is' if len(offenders) == 1 else 'are'} still unset "
            "or the default placeholder. Set real values for each, or set "
            "ALLOW_INSECURE=1 only for local development. Offending vars: "
            f"{', '.join(sorted(offenders))}."
        )


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

    # Separate Google OAuth app for Drive storage (optional).
    # Falls back to google_client_id / google_client_secret when not set,
    # so a single Google Cloud project can serve both sign-in and Drive.
    drive_google_client_id: Optional[str] = None
    drive_google_client_secret: Optional[str] = None

    # ---- personal-cloud storage (contributor's own Google Drive / OneDrive) ----
    # A contributor may keep their video bytes on their own cloud instead of the
    # server disk. The OAuth apps are registered by the operator; these are their
    # credentials. Leave a provider's client id empty and that provider is not
    # offered anywhere.
    onedrive_client_id: Optional[str] = None
    onedrive_client_secret: Optional[str] = None
    onedrive_tenant: str = "common"
    # AES key for encrypting OAuth refresh tokens at rest.
    # Generate with: python3 -c "import secrets,base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
    # REQUIRED when cloud storage is used. If empty, tokens are stored unencrypted
    # with a logged warning.
    cloud_token_key: Optional[str] = None
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
    # Passcode / share-link redemption is the public brute-force surface: a
    # visitor gets a bounded number of guesses, keyed per IP and target.
    public_redeem_rate_limit: int = 15
    public_redeem_rate_window: int = 60
    # Most recent live passcodes a single redemption verifies (bcrypt runs are
    # the cost unit: a wrong guess pays one per candidate, so this bounds the
    # CPU an unauthenticated caller can burn in one request).
    public_redeem_scan_limit: int = Field(default=100, ge=1, le=1000)
    # Where the rate limiter gets the caller's address (app.security.client_ip).
    # Default False: this process faces the client directly, so
    # `request.client.host` is the only address a caller cannot forge — any
    # client-supplied header (`X-Forwarded-For`...) is ignored. Set True ONLY
    # behind a same-host TLS-terminating proxy (the deploy compose's Caddy):
    # `request.client.host` is then the proxy for every visitor, and the real
    # client is read from the LAST hop of `X-Forwarded-For` — the one hop a
    # proxy on the same box wrote; every earlier hop was typed by the client.
    trust_proxy_headers: bool = False
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
    # The guard runs before the app (API *or* worker) can sign or accept
    # anything with a placeholder secret.
    validate_security_settings(s)
    s.ensure_dirs()
    return s


settings = get_settings()
