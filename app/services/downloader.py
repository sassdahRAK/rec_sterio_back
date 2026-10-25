"""yt-dlp download wrapper.

Only public / permitted content is downloaded. DRM, paywalled, private and
member-only videos are refused with a clear error - no bypass of any kind is
implemented or attempted.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

from app.config import settings
from app.security import detect_platform
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
    base.update(opts)
    return yt_dlp.YoutubeDL(base)


def download(
    source_url: str,
    video_id: UUID | str,
    on_progress: Optional[Callable[[int], None]] = None,
) -> DownloadResult:
    """Download the media into MEDIA_ROOT/{video_id}/media.<ext>."""
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

    ydl = _build_ydl(opts)
    try:
        with ydl:
            info = ydl.extract_info(source_url, download=True)
    except Exception as exc:  # yt-dlp raises DownloadError / ExtractorError
        message = str(exc)
        lowered = message.lower()
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
        raise TransientDownloadError(f"Download failed: {_first_line(message)}") from exc

    if not info:
        raise TransientDownloadError("Downloader returned no metadata.")

    path = _resolve_output_path(outdir, info)
    if path is None or not path.exists():
        raise TransientDownloadError("Download completed but no media file was produced.")

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
