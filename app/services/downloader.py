"""yt-dlp download wrapper.

Only public / permitted content is downloaded. DRM, paywalled, private and
member-only videos are refused with a clear error - no bypass of any kind is
implemented or attempted.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

from app.config import settings
from app.dns_pin import UnsafeHostError, pin_resolved_host
from app.errors import AppError
from app.security import assert_fetch_url_safe, detect_platform
from app.services import storage

logger = logging.getLogger("recsterio.downloader")

# Substrings that indicate the content is not publicly accessible.
PROTECTED_MARKERS = (
    "private video",
    "this video is private",
    "members-only",
    "members only",
    "join this channel",
    "sign in to confirm",
    "sign in to view",
    "login required",
    "paid members",
    "premium members",
    "drm",
    "protected video",
    "video unavailable",
    "this video is unavailable",
    "video is not available",
    "not available in your country",
    "account associated with this video has been terminated",
    "age-restricted",
    "confirm your age",
    "purchase",
    "rental",
)


class DownloadError(Exception):
    """Permanent failure (content is protected / not permitted)."""


class TransientDownloadError(Exception):
    """Transient failure - the job should be retried."""


@dataclass
class DownloadResult:
    path: Path
    title: Optional[str] = None
    duration_seconds: Optional[float] = None
    thumbnail_url: Optional[str] = None
    platform: str = "unknown"
    sha256: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    extractor: Optional[str] = None
    # TvNa collection schema (instruction.txt #6): fields the platform knows
    # better than a human retyping them.
    author_handle: Optional[str] = None
    caption: Optional[str] = None
    date_posted: Optional[str] = None


@dataclass
class ProgressReporter:
    """Throttles progress writes to the DB (at most every ~2 seconds)."""

    callback: Optional[Callable[[int], None]] = None
    interval: float = 2.0
    _last: float = field(default=0.0, repr=False)
    _last_value: int = field(default=-1, repr=False)

    def __call__(self, percent: int) -> None:
        if self.callback is None:
            return
        percent = max(0, min(100, int(percent)))
        now = time.monotonic()
        if percent == self._last_value and percent < 100:
            return
        if now - self._last < self.interval and percent < 100:
            return
        self._last = now
        self._last_value = percent
        try:
            self.callback(percent)
        except Exception:  # pragma: no cover - progress must never break the download
            logger.exception("progress callback failed")


def _build_ydl(opts: dict):
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise TransientDownloadError(
            "yt-dlp is not installed on the server (pip install -r requirements.txt)."
        ) from exc

    base = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "restrictfilenames": False,
        "retries": 3,
        "socket_timeout": 30,
        "consoletitle": False,
        # Explicitly refuse DRM / protected streams; we never attempt to bypass.
        "allow_unplayable_formats": False,
    }

    # Site bot checks (TikTok, YouTube) are the most common reason a public
    # link fails from a server IP. A cookies file exported by the operator from
    # a session they are entitled to use is the supported remedy; it is
    # configuration, never hardcoded, and never logged.
    if settings.ytdlp_cookies_file:
        cookies = Path(settings.ytdlp_cookies_file).expanduser()
        if not cookies.exists():
            # The path belongs in the operator's log, never in the chat error.
            logger.error("YTDLP_COOKIES_FILE points at %s, which does not exist.", cookies)
            raise DownloadError(
                "YTDLP_COOKIES_FILE is set but the cookies file is missing on the "
                "server. The operator must create it (see deploy docs)."
            )
        base["cookiefile"] = str(cookies)
    if settings.ytdlp_cookies_from_browser:
        browser, _, profile = settings.ytdlp_cookies_from_browser.partition(":")
        browser = browser.strip()
        profile = profile.strip() or None
        if not browser:
            raise DownloadError(
                "YTDLP_COOKIES_FROM_BROWSER is set but names no browser. "
                "Use e.g. 'chrome' or 'chrome:Profile 1'."
            )
        base["cookiesfrombrowser"] = (browser, profile, None, None)
    if settings.ytdlp_proxy:
        base["proxy"] = settings.ytdlp_proxy
    if settings.ytdlp_user_agent:
        base["user_agent"] = settings.ytdlp_user_agent
    if settings.ytdlp_impersonate and settings.ytdlp_impersonate.strip():
        # yt-dlp wants an ImpersonateTarget object here, not a plain string:
        # a string trips an isinstance assert inside yt-dlp as soon as the
        # impersonation handler (curl-cffi) is installed.
        try:
            base["impersonate"] = yt_dlp.networking.impersonate.ImpersonateTarget.from_str(
                settings.ytdlp_impersonate.strip()
            )
        except ValueError as exc:
            raise DownloadError(
                f"YTDLP_IMPERSONATE={settings.ytdlp_impersonate!r} is not a valid "
                "impersonation target. Use a browser name like 'chrome'."
            ) from exc

    base.update(opts)
    try:
        return yt_dlp.YoutubeDL(base)
    except yt_dlp.utils.YoutubeDLError as exc:
        # Construction happens outside download()'s try/except, so without this
        # wrap a missing curl-cffi would surface as an unclassified retry loop.
        raise DownloadError(
            f"Impersonation was requested (YTDLP_IMPERSONATE={settings.ytdlp_impersonate}) "
            "but its dependency is missing on the server. Install curl-cffi "
            f"(pip install -r requirements.txt). {_first_line(str(exc))}"
        ) from exc


def download(
    source_url: str,
    video_id: UUID | str,
    on_progress: Optional[Callable[[int], None]] = None,
    cookiefile: Optional[str] = None,
) -> DownloadResult:
    """Download the media into MEDIA_ROOT/{video_id}/media.<ext>.

    `cookiefile` is the requesting user's own cookies export (per-user
    session). When it exists it wins over every server-side session setting;
    yt-dlp would otherwise merge the operator's jar with the user's.

    The URL is re-validated here, at the fetch step, with the fail-closed
    guard (`assert_fetch_url_safe`): create-time validation happened long
    before this request runs, and a DNS record may have been flipped since.
    """
    try:
        assert_fetch_url_safe(source_url)
    except AppError as exc:
        # The guard raises AppError(422), which the worker does not translate
        # into a permanent job failure. A blocked fetch is permanent: report it
        # as such, with one user-safe line (no paths, no resolver internals).
        raise DownloadError(
            "The source URL resolves to a private or local network address, "
            "so the fetch was refused."
        ) from exc
    outdir = storage.video_dir(video_id)
    reporter = ProgressReporter(on_progress)

    def _hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                reporter(int(done * 100 / total))

    def _post_hook(d: dict) -> None:
        if d.get("status") == "finished":
            reporter(100)

    outtmpl = str(outdir / f"{storage.MEDIA_STEM}.%(ext)s")
    opts = {
        "outtmpl": {"default": outtmpl},
        "format": (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/bestvideo+bestaudio/best"
        ),
        "merge_output_format": "mp4",
        "progress_hooks": [_hook],
        "postprocessor_hooks": [_post_hook],
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
        ],
        "writethumbnail": False,
    }

    if cookiefile:
        path = Path(cookiefile)
        if path.exists():
            # Explicit override: yt-dlp MERGES cookiesfrombrowser with a
            # cookiefile, which would mix the operator's session into the
            # user's download. None disables the browser jar; proxy /
            # user_agent / impersonate still apply.
            opts["cookiefile"] = str(path)
            opts["cookiesfrombrowser"] = None
        else:
            # The row said there was a session, but the file is gone (e.g.
            # withdrawn between enqueue and run): fall back to server
            # defaults rather than failing the download outright.
            logger.warning(
                "per-user cookies file %s does not exist; downloading with "
                "server defaults",
                path,
            )

    try:
        with pin_resolved_host(source_url):
            # Constructing yt-dlp and fetching metadata both resolve the host,
            # so both run under the pin: the metadata request may only reach an
            # address that was public when the guard ran.
            ydl = _build_ydl(opts)
            try:
                with ydl:
                    info = ydl.extract_info(source_url, download=True)
            except Exception as exc:  # yt-dlp raises DownloadError / ExtractorError
                message = str(exc)
                lowered = message.lower()
                platform = detect_platform(source_url)
                if any(marker in lowered for marker in PROTECTED_MARKERS):
                    raise DownloadError(
                        "This content is protected, private or otherwise not publicly "
                        f"available: {_first_line(message)}"
                    ) from exc
                if "requested format is not available" in lowered:
                    raise TransientDownloadError(
                        "No usable media format was offered by the server. The content may "
                        "be protected or region-locked."
                    ) from exc
                # A 429 is the site throttling this server, not refusing it: retrying
                # later is the correct response, so check it before the permanent
                # bot-block markers.
                if any(marker in lowered for marker in RATE_LIMIT_MARKERS):
                    raise TransientDownloadError(
                        f"the site is rate-limiting this server; the job will retry with "
                        f"backoff: {_first_line(message)}"
                    ) from exc
                # A bot check is permanent for this configuration: retrying the same
                # request from the same IP produces the same wall. Say what fixes it.
                if any(marker in lowered for marker in BLOCK_MARKERS):
                    raise DownloadError(
                        f"{platform.capitalize() if platform != 'unknown' else 'The site'} "
                        f"blocked this request: {_first_line(message)}. {_block_hint(platform)}"
                    ) from exc
                raise TransientDownloadError(
                    f"Download failed: {_first_line(message)}"
                ) from exc
    except UnsafeHostError as exc:
        # The pin could not resolve a public address: same permanent, user-safe
        # refusal as a blocked pre-fetch guard.
        raise DownloadError(
            "The source URL resolves to a private or local network address, "
            "so the fetch was refused."
        ) from exc

    if not info:
        raise TransientDownloadError("Downloader returned no metadata.")

    path = _resolve_output_path(outdir, info)
    if path is None or not path.exists():
        raise TransientDownloadError("Download completed but no media file was produced.")

    # Redirect hops happen inside yt-dlp, so the final hop cannot be blocked
    # before it is taken. This is the "re-validate every hop" backstop: where
    # the fetch actually ended must also be a public address, otherwise the
    # bytes are discarded — a redirect must never smuggle private content in.
    final_url = str(info.get("url") or info.get("final_url") or "").strip()
    if final_url:
        try:
            assert_fetch_url_safe(final_url)
        except AppError as exc:
            try:
                path.unlink()
            except OSError:
                logger.warning("could not remove redirected-download output %s", path)
            raise DownloadError(
                "The download was redirected to a private or local network "
                "address, so the fetched file was discarded."
            ) from exc

    sha = storage.sha256_file(path)
    return DownloadResult(
        path=path,
        title=info.get("title"),
        duration_seconds=info.get("duration"),
        thumbnail_url=info.get("thumbnail"),
        platform=normalize_platform(info.get("extractor_key"), source_url),
        sha256=sha,
        file_size=storage.file_size(path),
        mime_type=storage.guess_mime(path),
        extractor=info.get("extractor_key"),
        author_handle=normalize_handle(info),
        caption=first_words(info.get("description") or info.get("title")),
        date_posted=normalize_upload_date(info.get("upload_date")),
    )


# ------------------------------------------------------------------ TvNa metadata
# The TvNa schema (instruction.txt #6) asks for an author handle, the first 20
# words of the caption verbatim, and a post date. yt-dlp already has all three
# in its info dict; formatting them here means the collector never retypes them.

CAPTION_WORDS = 20


def normalize_handle(info: dict) -> Optional[str]:
    """`@handle` when the platform exposes one, else the display name."""
    handle = (info.get("uploader_id") or "").strip()
    if handle:
        # TikTok/YouTube report the id without the @; Instagram with it.
        return handle if handle.startswith("@") else f"@{handle}"
    uploader = (info.get("uploader") or "").strip()
    return uploader or None


def first_words(text: Optional[str], limit: int = CAPTION_WORDS) -> Optional[str]:
    """The first `limit` words, verbatim — no ellipsis, no re-punctuation."""
    if not text:
        return None
    words = str(text).split()
    if not words:
        return None
    return " ".join(words[:limit])


def normalize_upload_date(value: Optional[str]) -> Optional[str]:
    """yt-dlp reports YYYYMMDD; the schema wants YYYY-MM-DD (or 'n.d.')."""
    if not value:
        return "n.d."
    raw = str(value).strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw or "n.d."


# The site is throttling this server; backing off is the remedy, not cookies.
RATE_LIMIT_MARKERS = (
    "too many requests",
    "rate limit",
    "http error 429",
)

# Errors that mean "the site refused this server", which no retry will fix.
BLOCK_MARKERS = (
    "unexpected response from webpage request",
    "captcha",
    "blocked",
    "bot check",
    "please verify",
    "confirm you're not a robot",
)

# What the operator should actually do, per platform. Retrying is not advice;
# supplying a session is.
COOKIE_HINTS = {
    "tiktok": (
        "TikTok refuses most requests that do not look like a real browser. "
        "Export a cookies file from a session you are entitled to use and set "
        "YTDLP_COOKIES_FILE. Also set YTDLP_IMPERSONATE=chrome so the request "
        "carries a real browser TLS fingerprint; cookies alone no longer get a "
        "server past the bot check."
    ),
    "facebook": (
        "Facebook shows almost nothing to an anonymous server. Export cookies "
        "from a logged-in session (c_user + xs) and set YTDLP_COOKIES_FILE. "
        "For local development, YTDLP_COOKIES_FROM_BROWSER (e.g. 'chrome') "
        "reads the same session straight from the browser instead."
    ),
    "instagram": (
        "Instagram requires a logged-in session for most media. Set "
        "YTDLP_COOKIES_FILE to a cookies file from a session you may use."
    ),
    "youtube": (
        "YouTube is asking this server to prove it is not a bot. Set "
        "YTDLP_COOKIES_FILE, or try YTDLP_PROXY from an address you control."
    ),
}


def _block_hint(platform: str) -> str:
    hint = COOKIE_HINTS.get(platform)
    if hint:
        return hint
    return (
        "The site refused this server. Setting YTDLP_COOKIES_FILE and "
        "YTDLP_IMPERSONATE is the supported remedy; retrying alone will not help."
    )


# The contract allows only these platform values.
CONTRACT_PLATFORMS = {"youtube", "tiktok", "facebook", "instagram", "direct", "unknown"}


def normalize_platform(extractor_key: str | None, source_url: str) -> str:
    """Map a yt-dlp extractor key onto the contract's platform enum.

    yt-dlp reports e.g. "generic" for a direct media URL, which is not one of
    the enum values the API contract allows; fall back to URL-based detection
    (which yields "direct" for a plain media link, "unknown" when there is no host).
    """
    key = (extractor_key or "").lower()
    if "youtube" in key:
        return "youtube"
    if "tiktok" in key:
        return "tiktok"
    if "facebook" in key or "fb" in key:
        return "facebook"
    if "instagram" in key:
        return "instagram"
    return detect_platform(source_url)


def _resolve_output_path(outdir: Path, info: dict) -> Optional[Path]:
    candidates: list[Path] = []
    for rd in info.get("requested_downloads") or []:
        fn = rd.get("filepath") or rd.get("_filename")
        if fn:
            candidates.append(Path(fn))
    for cand in candidates:
        if cand.exists():
            return cand
    found = storage.find_media_file(outdir.name)
    if found:
        return found
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.stem.startswith(storage.MEDIA_STEM):
            return p
    return None


def _first_line(message: str) -> str:
    for line in message.splitlines():
        line = line.strip()
        if line:
            return line[:400]
    return message[:400]
