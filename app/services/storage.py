"""Filesystem layout, checksums and cleanup for downloaded media."""
from __future__ import annotations

import hashlib
import mimetypes
import shutil
from pathlib import Path
from typing import Iterator, Optional
from uuid import UUID

from app.config import settings

AUDIO_FILENAME = "audio.wav"
MEDIA_STEM = "media"


def media_root() -> Path:
    root = settings.media_root_path
    root.mkdir(parents=True, exist_ok=True)
    return root


def video_dir(video_id: UUID | str) -> Path:
    d = media_root() / str(video_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    for p in sorted(d.iterdir()):
        if p.is_file() and p.stem == MEDIA_STEM and p.suffix.lower() != ".part":
            return p
    return None


def audio_file(video_id: UUID | str) -> Path:
    return video_dir(video_id) / AUDIO_FILENAME


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
    d = media_root() / str(video_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


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
