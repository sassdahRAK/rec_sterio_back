"""Filesystem layout, checksums and cleanup for downloaded media."""
from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Iterator, Optional
from uuid import UUID

from app.config import settings

AUDIO_FILENAME = "audio.wav"
AUDIO_MP3_FILENAME = "audio.mp3"
MEDIA_STEM = "media"
# Delivery-only copy for sending to a chat; never an ASR input.
COMPRESSED_STEM = "compressed"
# A locally cached frame. Platform CDN URLs are signed and expire, so the card
# is served from this file instead (see app.services.thumbnail).
THUMBNAIL_STEM = "thumbnail"
THUMBNAIL_EXTS = ("jpg", "jpeg", "png", "webp")
# A frame is decoration: anything larger than this is not a thumbnail.
THUMBNAIL_MAX_BYTES = 5 * 1024 * 1024


def media_root() -> Path:
    root = settings.media_root_path
    root.mkdir(parents=True, exist_ok=True)
    return root


def video_dir(video_id: UUID | str) -> Path:
    d = media_root() / str(video_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cookies_root() -> Path:
    """Per-user bot cookies dir; created on demand."""
    root = settings.cookies_root_path
    root.mkdir(parents=True, exist_ok=True)
    return root


def cookie_file(telegram_id: int) -> Path:
    """The Netscape cookies.txt one Telegram user uploaded via the bot."""
    return cookies_root() / f"{telegram_id}.txt"


def media_file(video_id: UUID | str, ext: str | None = None) -> Path:
    d = video_dir(video_id)
    if ext:
        return d / f"{MEDIA_STEM}.{ext.lstrip('.')}"
    existing = find_media_file(video_id)
    return existing or (d / f"{MEDIA_STEM}.mp4")


def find_media_file(video_id: UUID | str) -> Optional[Path]:
    d = media_root() / str(video_id)
    if not d.exists():
        return None
    candidates = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.stem == MEDIA_STEM and p.suffix.lower() not in (".part", ".cloud-tmp"):
            candidates.append(p)
    if candidates:
        return candidates[0]
    # Fall back to cloud-tmp only when no real file exists
    for p in sorted(d.iterdir()):
        if p.is_file() and p.stem == MEDIA_STEM and p.suffix.lower() == ".cloud-tmp":
            return p
    return None


def compressed_media_file(video_id: UUID | str) -> Path:
    """The delivery-only mp4 (a chat-sized copy of the original media)."""
    return video_dir(video_id) / f"{COMPRESSED_STEM}.mp4"


def find_compressed_file(video_id: UUID | str) -> Optional[Path]:
    """The compressed delivery copy if it exists on disk, else None."""
    path = media_root() / str(video_id) / f"{COMPRESSED_STEM}.mp4"
    if path.is_file():
        return path
    return None


def thumbnail_file(video_id: UUID | str) -> Optional[Path]:
    """The cached frame for this video, if one was saved, else None.

    The extension is chosen when the bytes are stored (see `save_thumbnail`),
    so this looks for each supported extension in a stable order.
    """
    d = media_root() / str(video_id)
    if not d.is_dir():
        return None
    for ext in THUMBNAIL_EXTS:
        path = d / f"{THUMBNAIL_STEM}.{ext}"
        if path.is_file():
            return path
    return None


def sniff_image_ext(data: bytes) -> Optional[str]:
    """The real image type from its magic bytes.

    CDNs routinely send an `application/octet-stream` (or wrong) Content-Type
    for a perfectly valid JPEG, so the bytes themselves decide the extension.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _declared_image_ext(content_type: str) -> Optional[str]:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    return {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/pjpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get(mime)


def save_thumbnail(video_id: UUID | str, data: bytes, content_type: str) -> Path:
    """Write the frame into the video's media dir, atomically.

    The extension comes from the bytes (sniffed) before the declared type, so a
    lying Content-Type cannot mislabel the file. More than
    `THUMBNAIL_MAX_BYTES` is refused: a giant payload is not a card thumbnail.
    Returns the stored path.
    """
    if not data:
        raise ValueError("thumbnail payload is empty")
    if len(data) > THUMBNAIL_MAX_BYTES:
        raise ValueError("thumbnail payload exceeds the 5 MB limit")
    ext = sniff_image_ext(data) or _declared_image_ext(content_type)
    if ext is None:
        raise ValueError("thumbnail payload is not a supported image")
    d = video_dir(video_id)
    target = d / f"{THUMBNAIL_STEM}.{ext}"
    tmp = d / f"{THUMBNAIL_STEM}.{ext}.tmp"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    # A re-cache in a different format must not leave the old frame behind:
    # `thumbnail_file` would keep returning it.
    for other in list(d.iterdir()):
        if other.is_file() and other.stem == THUMBNAIL_STEM and other != target:
            other.unlink(missing_ok=True)
    return target


def cloud_media_file(video_id: UUID | str) -> Path:
    """Temp local copy of a cloud-stored video, for the extract/transcribe pipeline.

    The authoritative bytes live in the contributor's drive; this file exists
    only while the pipeline needs a local path, and is deleted once audio.wav
    exists (see `runner._ensure_audio`).
    """
    return video_dir(video_id) / f"{MEDIA_STEM}.cloud-tmp"


def audio_file(video_id: UUID | str) -> Path:
    return video_dir(video_id) / AUDIO_FILENAME


def audio_mp3_file(video_id: UUID | str) -> Path:
    """The .mp3 listening copy (not an ASR artifact — that is audio.wav)."""
    return video_dir(video_id) / AUDIO_MP3_FILENAME


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def delete_video_files(video_id: UUID | str) -> None:
    # Removes the whole directory, so the cached thumbnail (THUMBNAIL_STEM)
    # and every other derived artefact go with it; nothing extra to clean up.
    d = media_root() / str(video_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def clone_video_files(source_id: UUID | str, dest_id: UUID | str) -> int:
    """Populate a copy's media dir with links to the source's files.

    Hard links, not byte copies: both directory entries point at the same
    inode, so a fork costs no disk and no re-download. Deletion stays correct
    without reference counting, because `delete_video_files` removes a
    directory and only that directory's link — the other row keeps its own.
    Across filesystems `os.link` raises EXDEV, so fall back to a real copy.

    Returns how many files were linked.
    """
    src = media_root() / str(source_id)
    if not src.is_dir():
        return 0
    dest = video_dir(dest_id)
    linked = 0
    for entry in sorted(src.iterdir()):
        if not entry.is_file() or entry.suffix.lower() == ".part":
            continue
        target = dest / entry.name
        if target.exists():
            continue
        try:
            os.link(entry, target)
        except OSError:
            shutil.copy2(entry, target)
        linked += 1
    return linked


def cloud_storage_available(video) -> bool:
    """A cloud-stored unit is readable only while its drive link is live."""
    if video.storage_backend is None:
        return True
    if not video.drive_file_id or video.drive_link_id is None:
        return False
    link = video.drive_link
    return link is not None and bool(link.refresh_token)


def iter_file(path: Path, start: int = 0, end: Optional[int] = None, chunk_size: int = 512 * 1024) -> Iterator[bytes]:
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = None if end is None else (end - start + 1)
        while True:
            if remaining is not None and remaining <= 0:
                break
            size = chunk_size if remaining is None else min(chunk_size, remaining)
            chunk = fh.read(size)
            if not chunk:
                break
            if remaining is not None:
                remaining -= len(chunk)
            yield chunk
