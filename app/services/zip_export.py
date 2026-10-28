"""ZIP export of a dataset folder (instruction.txt #5, #6).

Layout mirrors the folder tree so a download is usable without re-organising it:

    <folder>/<unit>/video/<media>
    <folder>/<unit>/audio/audio.wav
    <folder>/<unit>/transcript/<unit>.txt
    <folder>/<sub-folder>/<unit>/...

A "unit" is one video. Sub-folders nest, which is what makes a many-video
dataset downloadable as a single archive.
"""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Folder, Transcript, Video
from app.services import storage

logger = logging.getLogger("recsterio.zip")

# Guard rails: a pathological tree must not produce an unbounded archive.
MAX_ENTRIES = 5000
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
TRANSCRIPT_EXCERPT_CHARS = 4000


def safe_name(name: Optional[str], fallback: str) -> str:
    """A path component that is safe on macOS, Linux and Windows."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name or "").strip()
    cleaned = cleaned.strip(". ")
    return cleaned[:120] or fallback


def video_slug(video: Video) -> str:
    """Stable, readable unit directory name that never collides."""
    base = safe_name(video.title, "video")
    return f"{base}-{str(video.id)[:8]}"


def current_transcript(db: Session, video_id: UUID) -> Optional[Transcript]:
    """The transcript a card should show: newest ready version, else newest."""
    ready = db.execute(
        select(Transcript)
        .where(Transcript.video_id == video_id, Transcript.status == "ready")
        .order_by(Transcript.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    if ready is not None:
        return ready
    return db.execute(
        select(Transcript)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.version.desc())
        .limit(1)
    ).scalar_one_or_none()


def transcript_text(transcript: Optional[Transcript]) -> str:
    """Plain-text transcript for the archive.

    Segments are preferred over the flat `content` field because they keep the
    per-segment Khmer/English language tags, which is the whole point of the
    dataset.
    """
    if transcript is None:
        return ""
    if transcript.segments:
        lines = []
        for seg in transcript.segments:
            lang = (seg.get("language") or "und").lower()
            text = str(seg.get("text", "")).strip()
            if text:
                lines.append(f"[{lang}] {text}")
        if lines:
            return "\n".join(lines)
    return (transcript.content or "").strip()


def build_folder_zip(db: Session, folder: Folder, dest: Path) -> int:
    """Write `folder` and its descendants to `dest`. Returns the entry count."""
    written = 0
    total_bytes = 0

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:

        def add(path_in_zip: str, source: Path) -> None:
            nonlocal written, total_bytes
            if written >= MAX_ENTRIES:
                raise ValueError("archive would exceed the entry limit")
            size = source.stat().st_size
            if total_bytes + size > MAX_TOTAL_BYTES:
                raise ValueError("archive would exceed the size limit")
            zf.write(source, path_in_zip)
            written += 1
            total_bytes += size

        def walk(node: Folder, prefix: str) -> None:
            nonlocal written, total_bytes
            videos = db.execute(
                select(Video)
                .where(Video.folder_id == node.id)
                .order_by(Video.position.asc(), Video.created_at.asc())
            ).scalars().all()

            for video in videos:
                unit = f"{prefix}/{video_slug(video)}"
                media = storage.find_media_file(video.id)
                if media is not None and media.exists():
                    add(f"{unit}/video/{media.name}", media)
                audio = storage.audio_file(video.id)
                if audio.exists():
                    add(f"{unit}/audio/{storage.AUDIO_FILENAME}", audio)
                transcript = current_transcript(db, video.id)
                text = transcript_text(transcript)
                if text:
                    # Written from memory; a transcript is small and text-only.
                    # The entry cap applies here too — a tree of transcript-only
                    # units must not exceed MAX_ENTRIES via this branch.
                    if written >= MAX_ENTRIES:
                        raise ValueError("archive would exceed the entry limit")
                    info = zipfile.ZipInfo(f"{unit}/transcript/{video_slug(video)}.txt")
                    payload = text.encode("utf-8")
                    if total_bytes + len(payload) > MAX_TOTAL_BYTES:
                        raise ValueError("archive would exceed the size limit")
                    zf.writestr(info, payload)
                    written += 1
                    total_bytes += len(payload)

            children = db.execute(
                select(Folder)
                .where(Folder.parent_id == node.id)
                .order_by(Folder.position.asc(), Folder.created_at.asc())
            ).scalars().all()
            for child in children:
                walk(child, f"{prefix}/{safe_name(child.name, str(child.id)[:8])}")

        walk(folder, safe_name(folder.name, str(folder.id)[:8]))

    if written == 0:
        raise ValueError("folder has no downloadable media")
    return written
