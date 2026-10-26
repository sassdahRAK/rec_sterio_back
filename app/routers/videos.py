"""Video routes: list, create, detail, update, delete, media streaming, retry."""
from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.access import require_manage, require_read, require_video, visible_folders
from app.auth import Principal, get_actor, get_optional_principal, get_principal
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import Folder, Job, Merge, Transcript, Video
from app.schemas import (
    JobOut,
    TranscriptOut,
    VideoCreate,
    VideoDetailOut,
    VideoListOut,
    VideoOut,
    VideoUpdate,
)
from app.security import detect_platform, enforce_rate_limit, validate_source_url
from app.services import asr, storage
from app.worker import queue

logger = logging.getLogger("recsterio.videos")

router = APIRouter(prefix="/videos", tags=["videos"])

SORTABLE = {"created_at", "-created_at", "title", "-title"}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_video_or_404(db: Session, video_id: UUID) -> Video:
    video = db.get(Video, video_id)
    if video is None:
        raise AppError(404, "not_found", "Video not found.", [{"id": str(video_id)}])
    return video


@router.get("", response_model=VideoListOut)
def list_videos(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    platform: Optional[str] = None,
    q: Optional[str] = None,
    folder_id: Optional[UUID] = None,
    sort: str = Query("-created_at"),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoListOut:
    stmt = select(Video)
    count_stmt = select(func.count()).select_from(Video)

    # The console's library is only fully visible to the operator. A member
    # sees the folders shared with them (plus what they ingested themselves),
    # so a folder's privacy is not merely cosmetic.
    if principal.role not in ("admin", "telegram"):
        roles = visible_folders(db, principal)
        conditions = [Video.created_by == principal.actor]
        if roles:
            conditions.append(Video.folder_id.in_(list(roles.keys())))
        stmt = stmt.where(or_(*conditions))
        count_stmt = count_stmt.where(or_(*conditions))
    if folder_id is not None:
        stmt = stmt.where(Video.folder_id == folder_id)
        count_stmt = count_stmt.where(Video.folder_id == folder_id)
    if status:
        stmt = stmt.where(Video.status == status)
        count_stmt = count_stmt.where(Video.status == status)
    if platform:
        stmt = stmt.where(Video.platform == platform)
        count_stmt = count_stmt.where(Video.platform == platform)
    if q:
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(or_(Video.title.ilike(pattern), Video.source_url.ilike(pattern)))
        count_stmt = count_stmt.where(
            or_(Video.title.ilike(pattern), Video.source_url.ilike(pattern))
        )
    total = db.execute(count_stmt).scalar_one()
    if sort not in SORTABLE:
        raise AppError(
            422,
            "validation_failed",
            "sort must be one of created_at, -created_at, title.",
            [{"field": "sort", "issue": "unsupported_value", "allowed": sorted(SORTABLE)}],
        )
    column = {
        "created_at": Video.created_at,
        "-created_at": Video.created_at.desc(),
        "title": Video.title,
        "-title": Video.title.desc(),
    }[sort]
    stmt = stmt.order_by(column, Video.id).offset((page - 1) * page_size).limit(page_size)
    items = db.execute(stmt).scalars().all()
    return VideoListOut(
        items=[VideoOut.model_validate(v) for v in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=VideoOut, status_code=201)
def create_video(
    payload: VideoCreate,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    enforce_rate_limit(
        f"video_create:{_client_ip(request)}",
        settings.video_create_rate_limit,
        settings.video_create_rate_window,
    )
    source_url = validate_source_url(payload.source_url)
    if payload.folder_id is not None:
        folder = db.get(Folder, payload.folder_id)
        if folder is None:
            raise AppError(404, "not_found", "Folder not found.", [{"id": str(payload.folder_id)}])
        # Ingesting *into* a folder is an edit of that dataset.
        require_manage(db, folder, principal)

    # Idempotency: the same URL (that is not failed) must not be duplicated.
    existing = db.execute(
        select(Video).where(Video.source_url == source_url, Video.status != "failed")
    ).scalars().first()
    if existing is not None:
        raise AppError(
            409,
            "conflict",
            "A video with this source_url already exists.",
            [
                {
                    "field": "source_url",
                    "issue": "duplicate_video",
                    "video_id": str(existing.id),
                }
            ],
            # The existing resource is returned alongside the error envelope so
            # clients do not have to create a duplicate.
            extra={"video": VideoOut.model_validate(existing).model_dump(mode="json")},
        )

    position = 0
    if payload.folder_id is not None:
        position = db.execute(
            select(func.coalesce(func.max(Video.position), -1) + 1).where(
                Video.folder_id == payload.folder_id
            )
        ).scalar_one()

    video = Video(
        source_url=source_url,
        platform=detect_platform(source_url),
        title=payload.title,
        notes=payload.notes,
        folder_id=payload.folder_id,
        position=position,
        status="pending",
        progress=0,
        created_by=principal.actor,
    )
    db.add(video)
    db.flush()

    queue.enqueue(
        db,
        "download",
        video_id=video.id,
        payload={"chain": ["extract_audio", "transcribe"] if payload.start_transcription else []},
        priority=100,
    )
    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)


ALLOWED_UPLOAD_SUFFIXES = {
    ".mp4",
    ".m4v",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
}


def _safe_suffix(filename: str | None) -> str:
    """Allowed lowercase suffix from a client-supplied filename, or ''."""
    if not filename:
        return ""
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in ALLOWED_UPLOAD_SUFFIXES else ""


@router.post("/upload", response_model=VideoOut, status_code=201)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    folder_id: Optional[UUID] = Form(None),
    start_transcription: bool = Form(True),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    """Ingest a media file the operator already has on disk.

    Ingestion must not depend on a platform serving a download: TikTok and
    YouTube refuse most server-side requests, so an operator who has the file
    can upload it directly and get the same pipeline (checksum, audio
    extraction, transcription) behind it.
    """
    enforce_rate_limit(
        f"video_create:{_client_ip(request)}",
        settings.video_create_rate_limit,
        settings.video_create_rate_window,
    )

    suffix = _safe_suffix(file.filename)
    if not suffix:
        raise AppError(
            415,
            "unsupported_media_type",
            "Unsupported file type. Upload an audio or video file.",
            [
                {
                    "field": "file",
                    "issue": "unsupported_type",
                    "filename": file.filename,
                    "allowed": sorted(ALLOWED_UPLOAD_SUFFIXES),
                }
            ],
        )

    limit_bytes = settings.max_upload_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit_bytes:
        raise AppError(
            413,
            "payload_too_large",
            f"File exceeds the {settings.max_upload_mb} MB upload limit.",
            [{"limit_bytes": limit_bytes}],
        )

    if folder_id is not None:
        folder = db.get(Folder, folder_id)
        if folder is None:
            raise AppError(404, "not_found", "Folder not found.", [{"id": str(folder_id)}])
        require_manage(db, folder, principal)

    video = Video(
        source_url=f"upload://{Path(file.filename or 'upload').name}",
        platform="direct",
        title=title or None,
        notes=notes or None,
        folder_id=folder_id,
        status="downloading",
        progress=0,
        created_by=principal.actor,
    )
    db.add(video)
    db.flush()

    dest = storage.media_file(video.id, suffix)
    digest = hashlib.sha256()
    written = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit_bytes:
                    raise AppError(
                        413,
                        "payload_too_large",
                        f"File exceeds the {settings.max_upload_mb} MB upload limit.",
                        [{"limit_bytes": limit_bytes}],
                    )
                digest.update(chunk)
                out.write(chunk)
    except AppError:
        _discard_upload(db, video, dest)
        raise
    except OSError as exc:
        _discard_upload(db, video, dest)
        raise AppError(
            500, "upload_failed", "Could not write the uploaded file to disk."
        ) from exc

    if written == 0:
        _discard_upload(db, video, dest)
        raise AppError(
            422, "validation_failed", "The uploaded file is empty.", [{"field": "file"}]
        )

    checksum = digest.hexdigest()
    # Idempotency: the same bytes are the same asset, whatever the filename.
    duplicate = db.execute(
        select(Video).where(Video.checksum_sha256 == checksum, Video.status != "failed")
    ).scalars().first()
    if duplicate is not None:
        _discard_upload(db, video, dest)
        raise AppError(
            409,
            "conflict",
            "This file has already been uploaded.",
            [{"field": "file", "issue": "duplicate_video", "video_id": str(duplicate.id)}],
            extra={"video": VideoOut.model_validate(duplicate).model_dump(mode="json")},
        )

    video.checksum_sha256 = checksum
    video.file_size = written
    video.local_path = str(dest)
    # curl and many clients send application/octet-stream for any file, which
    # tells the dashboard's <video> element nothing. Fall back to the suffix.
    declared_type = (file.content_type or "").strip()
    if not declared_type or declared_type == "application/octet-stream":
        video.mime_type = storage.guess_mime(dest)
    else:
        video.mime_type = declared_type
    video.duration_seconds = asr.probe_duration(dest)
    video.status = "downloaded"
    video.progress = 100

    db.flush()
    queue.enqueue(
        db,
        "extract_audio",
        video_id=video.id,
        payload={"chain": ["transcribe"] if start_transcription else []},
        priority=100,
    )
    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)


def _discard_upload(db: Session, video: Video, dest: Path) -> None:
    """Remove a half-created upload so a rejected request leaves nothing behind."""
    try:
        if dest.exists():
            dest.unlink()
    except OSError:
        logger.warning("could not remove rejected upload at %s", dest)
    db.rollback()


@router.get("/{video_id}", response_model=VideoDetailOut)
def get_video(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> VideoDetailOut:
    video = get_video_or_404(db, video_id)
    # A video is as readable as the folder it lives in (instruction.txt #3, #9).
    require_video(db, video, principal)
    transcripts = db.execute(
        select(Transcript)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.version.desc())
    ).scalars().all()
    out = VideoDetailOut.model_validate(video)
    out.transcripts = [TranscriptOut.model_validate(t) for t in transcripts]
    # Job history is operational: only signed-in users see the queue internals.
    if principal.is_signed_in or principal.role == "telegram":
        jobs = db.execute(
            select(Job).where(Job.video_id == video_id).order_by(Job.created_at.desc())
        ).scalars().all()
        out.jobs = [JobOut.model_validate(j) for j in jobs]
    return out


@router.patch("/{video_id}", response_model=VideoOut)
def update_video(
    video_id: UUID,
    payload: VideoUpdate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    video = get_video_or_404(db, video_id)
    require_video(db, video, principal, need="editor")
    data = payload.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(video, field, value)
    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)


@router.delete("/{video_id}", status_code=204)
def delete_video(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> Response:
    video = get_video_or_404(db, video_id)
    require_video(db, video, principal, need="editor")
    db.execute(Transcript.__table__.delete().where(Transcript.video_id == video_id))
    db.execute(Merge.__table__.delete().where(Merge.video_id == video_id))
    db.execute(Job.__table__.delete().where(Job.video_id == video_id))
    db.delete(video)
    db.commit()
    storage.delete_video_files(video_id)
    return Response(status_code=204)


@router.post("/{video_id}/retry", response_model=JobOut, status_code=202)
def retry_video(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> JobOut:
    video = get_video_or_404(db, video_id)
    require_video(db, video, principal, need="editor")
    job = queue.enqueue(
        db,
        "download",
        video_id=video.id,
        payload={"chain": ["extract_audio", "transcribe"]},
        priority=50,
    )
    if video.status in ("failed", "pending", "quarantined"):
        video.status = "pending"
        video.error = None
        video.progress = 0
    db.commit()
    db.refresh(job)
    return JobOut.model_validate(job)


# ------------------------------------------------------------------ media


def _parse_range(header: str, size: int) -> tuple[int, int]:
    """Parse a single-range `Range: bytes=...` header. Raises ValueError when invalid."""
    spec = header.strip()
    if not spec.lower().startswith("bytes="):
        raise ValueError("unsupported range unit")
    spec = spec.split("=", 1)[1].strip()
    if "," in spec:
        raise ValueError("multiple ranges are not supported")
    if "-" not in spec:
        raise ValueError("malformed range")
    start_raw, end_raw = spec.split("-", 1)
    start_raw, end_raw = start_raw.strip(), end_raw.strip()
    if start_raw == "":
        # suffix range: last N bytes
        if end_raw == "":
            raise ValueError("malformed range")
        length = int(end_raw)
        if length <= 0:
            raise ValueError("malformed range")
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else size - 1
    if start > end or start >= size:
        raise ValueError("range not satisfiable")
    end = min(end, size - 1)
    return start, end


def _authorize_media(db: Session, video: Video, principal) -> None:
    """Gate media on the folder it lives in, not just on knowing the UUID.

    The public portal serves cards from folders, so a video's media is as
    public as its folder. An unfiled video is console-only.
    """
    if video.folder_id is None:
        if not principal.is_signed_in:
            raise AppError(404, "not_found", "Media is not available yet for this video.")
        return
    folder = db.get(Folder, video.folder_id)
    if folder is not None:
        require_read(db, folder, principal)


@router.get("/{video_id}/media")
def stream_media(
    video_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal=Depends(get_optional_principal),
) -> Response:
    video = get_video_or_404(db, video_id)
    _authorize_media(db, video, principal)
    path = storage.find_media_file(video_id)
    if path is None or not path.exists():
        raise AppError(404, "not_found", "Media is not available yet for this video.")
    size = path.stat().st_size
    mime = video.mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    range_header = request.headers.get("range")
    if not range_header:
        return StreamingResponse(
            storage.iter_file(path, 0, size - 1),
            media_type=mime,
            headers={
                "Content-Length": str(size),
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'inline; filename="{path.name}"',
            },
        )

    try:
        start, end = _parse_range(range_header, size)
    except ValueError:
        return JSONResponse(
            status_code=416,
            content={
                "error": {
                    "code": "range_not_satisfiable",
                    "message": "The requested range is not satisfiable.",
                    "details": [{"size": size, "range": range_header}],
                }
            },
            headers={"Content-Range": f"bytes */{size}"},
        )

    return StreamingResponse(
        storage.iter_file(path, start, end),
        status_code=206,
        media_type=mime,
        headers={
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{path.name}"',
        },
    )


@router.get("/{video_id}/audio")
def download_audio(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal=Depends(get_optional_principal),
) -> Response:
    """The extracted 16 kHz WAV, for download from a public data card."""
    video = get_video_or_404(db, video_id)
    _authorize_media(db, video, principal)
    path = storage.audio_file(video_id)
    if not path.exists():
        raise AppError(404, "not_found", "Audio has not been extracted for this video.")
    return StreamingResponse(
        storage.iter_file(path),
        media_type="audio/wav",
        headers={
            "Content-Length": str(path.stat().st_size),
            "Content-Disposition": f'attachment; filename="{video_id}-audio.wav"',
        },
    )
