"""The local frame cache behind a video card.

Platform CDN thumbnail URLs are signed with a short expiry (TikTok's
`p16-common-sign.tiktokcdn.com` links are the usual offender): once the
signature ages out the URL answers `403 text/html` and the browser shows a
placeholder with no console error. The fix is to keep the bytes on disk in the
video's media directory and serve them from the API, so a card keeps rendering
for as long as the row exists.

Nothing here may fail an ingest: every entry point returns None on error.
"""
from __future__ import annotations

import logging
import mimetypes
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.orm import object_session

from app.errors import AppError
from app.models import Folder
from app.security import assert_fetch_url_safe
from app.services import storage
from app.services.http_client import http_bytes

logger = logging.getLogger("recsterio.thumbnail")

# A browser-like UA: tiktokcdn (and most CDNs) reject the httpx default.
FETCH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# The download path budgets `socket_timeout: 30` with 3 retries; a re-derive
# that outlives that same budget is a failure, not something to wait on.
DERIVE_TIMEOUT_SECONDS = 120.0

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _header(headers: dict, name: str) -> str:
    """Case-insensitive lookup across a plain response-header dict."""
    wanted = name.lower()
    for key, value in (headers or {}).items():
        if str(key).lower() == wanted:
            return str(value)
    return ""


def cache_from_url(video_id: UUID | str, url: Optional[str]) -> Optional[Path]:
    """Fetch `url` once and keep the bytes as this video's cached frame.

    A thumbnail is decoration, so every failure returns None and is logged for
    the operator; this function never raises.
    """
    if not url:
        return None
    try:
        assert_fetch_url_safe(url)
    except Exception as exc:  # AppError: the SSRF guard refused it
        logger.warning("thumbnail URL refused for %s: %s", video_id, exc)
        return None
    try:
        status, headers, body = http_bytes(
            "GET", url, headers={"User-Agent": FETCH_USER_AGENT}
        )
    except Exception as exc:
        logger.warning("thumbnail fetch failed for %s: %s", video_id, exc)
        return None
    if status != 200 or not body:
        logger.warning("thumbnail fetch for %s returned HTTP %s", video_id, status)
        return None
    try:
        return storage.save_thumbnail(
            video_id, body, _header(headers, "content-type")
        )
    except Exception as exc:
        logger.warning("thumbnail save failed for %s: %s", video_id, exc)
        return None


def derive_for_video(source_url: Optional[str]) -> Optional[tuple[bytes, str]]:
    """Re-derive a fresh frame for a page whose stored URL has expired.

    The stored `thumbnail_url` embeds a signature that cannot be refreshed in
    place, so the page is re-extracted with yt-dlp (the same construction and
    cookie/impersonation/proxy settings the download path uses). yt-dlp writes
    the thumbnail into a temp dir; the file is read back. Bounded by
    `DERIVE_TIMEOUT_SECONDS`, and any failure returns None.
    """
    if not source_url:
        return None
    try:
        return _run_bounded(lambda: _derive_blocking(source_url))
    except Exception as exc:
        logger.warning("thumbnail re-derive failed: %s", exc)
        return None


def _run_bounded(fn, timeout: float = DERIVE_TIMEOUT_SECONDS):
    """Run `fn` on a daemon thread and give up after `timeout` seconds.

    A daemon thread, unlike ThreadPoolExecutor's workers, cannot make a
    timed-out extraction keep the process alive.
    """
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller
            outcome["error"] = exc

    worker = threading.Thread(target=_target, name="thumbnail-derive", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError("thumbnail re-derive timed out")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _derive_blocking(source_url: str) -> Optional[tuple[bytes, str]]:
    from app.dns_pin import pin_resolved_host
    from app.services.downloader import _build_ydl

    tmpdir = Path(tempfile.mkdtemp(prefix="recsterio-thumb-"))
    try:
        opts = {
            # Write the thumbnail but not the media: yt-dlp applies the same
            # cookies / impersonation / proxy context it would for a download.
            "skip_download": True,
            "writethumbnail": True,
            "overwrites": True,
            "outtmpl": str(tmpdir / "thumb.%(ext)s"),
        }
        with pin_resolved_host(source_url):
            ydl = _build_ydl(opts)
            with ydl:
                info = ydl.extract_info(source_url, download=True) or {}
        written = [
            p
            for p in tmpdir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        ]
        if written:
            path = max(written, key=lambda p: p.stat().st_size)
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            return path.read_bytes(), mime
        # Some extractors only list URLs and write nothing. Fetch the best one
        # directly as a last resort.
        url = _best_thumbnail_url(info)
        if not url:
            return None
        try:
            assert_fetch_url_safe(url)
        except AppError:
            return None
        status, headers, body = http_bytes(
            "GET", url, headers={"User-Agent": FETCH_USER_AGENT}
        )
        if status != 200 or not body:
            return None
        return body, _header(headers, "content-type") or "image/jpeg"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _best_thumbnail_url(info: dict) -> Optional[str]:
    """The highest-preference, largest listed thumbnail URL."""
    thumbs = [
        t for t in (info.get("thumbnails") or []) if isinstance(t, dict) and t.get("url")
    ]
    if thumbs:
        def score(t: dict) -> tuple[int, int, int]:
            preference = t.get("preference")
            return (
                0 if preference is None else int(preference),
                int(t.get("width") or 0),
                int(t.get("height") or 0),
            )

        return str(max(thumbs, key=score)["url"])
    return info.get("thumbnail") or None


def _folder_for(video) -> Optional[Folder]:
    """The video's folder, read through whichever session owns the instance."""
    if video.folder_id is None:
        return None
    db = object_session(video)
    if db is None:
        return None
    return db.get(Folder, video.folder_id)


def public_thumbnail_url(video, folder: Optional[Folder] = None) -> Optional[str]:
    """What a card's `thumbnail_url` should be.

    The local endpoint wins only when a frame is cached AND the video sits in
    a folder a visitor can read — the same test `public._public_tree` applies
    (the folder itself is published). Everything else keeps the stored CDN
    URL, so private rows behave exactly as they did before.
    """
    if storage.thumbnail_file(video.id) is None:
        return video.thumbnail_url
    if folder is None:
        folder = _folder_for(video)
    if folder is not None and folder.visibility == "public":
        return f"/api/v1/public/videos/{video.id}/thumbnail"
    return video.thumbnail_url
