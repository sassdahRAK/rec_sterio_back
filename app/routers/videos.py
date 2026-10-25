"""Video routes: list, create, detail, update, delete, media streaming, retry."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.auth import get_actor
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import Job, Merge, Transcript, Video
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
from app.services import storage
from app.worker import queue

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
    sort: str = Query("-created_at"),
    db: Session = Depends(get_db),
    actor: str = Depends(get_actor),
) -> VideoListOut:
    stmt = select(Video)
    count_stmt = select(func.count()).select_from(Video)
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
    actor: str = Depends(get_actor),
) -> VideoOut:
    enforce_rate_limit(
        f"video_create:{_client_ip(request)}",
        settings.video_create_rate_limit,
        settings.video_create_rate_window,
    )
    source_url = validate_source_url(payload.source_url)

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

    video = Video(
        source_url=source_url,
        platform=detect_platform(source_url),
        title=payload.title,
        notes=payload.notes,
        status="pending",
        progress=0,
        created_by=actor,
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


@router.get("/{video_id}", response_model=VideoDetailOut)
def get_video(
    video_id: UUID,
    db: Session = Depends(get_db),
    actor: str = Depends(get_actor),
) -> VideoDetailOut:
    video = get_video_or_404(db, video_id)
    transcripts = db.execute(
        select(Transcript)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.version.desc())
    ).scalars().all()
    jobs = db.execute(
        select(Job).where(Job.video_id == video_id).order_by(Job.created_at.desc())
    ).scalars().all()
    out = VideoDetailOut.model_validate(video)
    out.transcripts = [TranscriptOut.model_validate(t) for t in transcripts]
    out.jobs = [JobOut.model_validate(j) for j in jobs]
    return out


@router.patch("/{video_id}", response_model=VideoOut)
def update_video(
    video_id: UUID,
    payload: VideoUpdate,
    db: Session = Depends(get_db),
    actor: str = Depends(get_actor),
) -> VideoOut:
    video = get_video_or_404(db, video_id)
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
    actor: str = Depends(get_actor),
) -> Response:
    video = get_video_or_404(db, video_id)
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
    actor: str = Depends(get_actor),
) -> JobOut:
    video = get_video_or_404(db, video_id)
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


@router.get("/{video_id}/media")
def stream_media(
    video_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
    actor: str = Depends(get_actor),
) -> Response:
    video = get_video_or_404(db, video_id)
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
