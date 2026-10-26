"""Transcript routes: current transcript, version history, manual edit, generation."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.access import require_video
from app.auth import Principal, get_optional_principal, get_principal
from app.db import get_db
from app.errors import AppError
from app.models import Transcript, Video
from app.schemas import (
    GenerateRequest,
    GenerateResponse,
    JobOut,
    SegmentList,
    TranscriptCleanedUpdate,
    TranscriptListOut,
    TranscriptOut,
    TranscriptUpdate,
)
from app.worker import queue

router = APIRouter(prefix="/videos", tags=["transcripts"])


def _get_video(db: Session, video_id: UUID) -> Video:
    video = db.get(Video, video_id)
    if video is None:
        raise AppError(404, "not_found", "Video not found.", [{"id": str(video_id)}])
    return video


def _current_transcript(db: Session, video_id: UUID) -> Optional[Transcript]:
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


def _next_version(db: Session, video_id: UUID) -> int:
    current = db.execute(
        select(Transcript.version)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    return (current or 0) + 1


@router.get("/{video_id}/transcript", response_model=TranscriptOut)
def get_transcript(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> TranscriptOut:
    video = _get_video(db, video_id)
    # Public data cards show their transcript (instruction.txt #3, #8), so this
    # read follows the video's folder access rather than requiring a sign-in.
    require_video(db, video, principal)
    transcript = _current_transcript(db, video_id)
    if transcript is None:
        raise AppError(404, "not_found", "No transcript exists for this video.")
    return TranscriptOut.model_validate(transcript)


@router.get("/{video_id}/transcripts", response_model=TranscriptListOut)
def list_transcripts(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> TranscriptListOut:
    require_video(db, _get_video(db, video_id), principal)
    items = db.execute(
        select(Transcript)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.version.desc())
    ).scalars().all()
    return TranscriptListOut(items=[TranscriptOut.model_validate(t) for t in items])


@router.put("/{video_id}/transcript", response_model=TranscriptOut, status_code=201)
def update_transcript(
    video_id: UUID,
    payload: TranscriptUpdate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> TranscriptOut:
    """Manual transcript edit - always creates a NEW version, never mutates history.

    A human correction is what turns a Raw card into a Cleaned card
    (instruction.txt #8), so a manual edit marks the new version cleaned unless
    the caller says otherwise.
    """
    video = _get_video(db, video_id)
    require_video(db, video, principal, need="editor")
    if payload.content is None or not payload.content.strip():
        raise AppError(
            422,
            "validation_failed",
            "content must not be empty.",
            [{"field": "content", "issue": "empty"}],
        )
    segments = payload.segments
    if segments is None:
        existing = _current_transcript(db, video_id)
        segments = existing.segments if existing is not None else None
    transcript = Transcript(
        video_id=video_id,
        version=_next_version(db, video_id),
        source="manual",
        language=payload.language,
        content=payload.content,
        segments=segments,
        status="ready",
        cleaned=True if payload.cleaned is None else payload.cleaned,
        created_by=principal.actor,
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)
    return TranscriptOut.model_validate(transcript)


@router.patch("/{video_id}/transcript", response_model=TranscriptOut)
def set_transcript_cleaned(
    video_id: UUID,
    payload: TranscriptCleanedUpdate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> TranscriptOut:
    """Flag the current version as Cleaned (or back to Raw) without a new version."""
    video = _get_video(db, video_id)
    require_video(db, video, principal, need="editor")
    transcript = _current_transcript(db, video_id)
    if transcript is None:
        raise AppError(404, "not_found", "No transcript exists for this video.")
    transcript.cleaned = payload.cleaned
    db.commit()
    db.refresh(transcript)
    return TranscriptOut.model_validate(transcript)


@router.post("/{video_id}/transcript/generate", response_model=GenerateResponse, status_code=202)
def generate_transcript(
    video_id: UUID,
    payload: GenerateRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> GenerateResponse:
    """Enqueue transcription work (whisper / gemini / both)."""
    video = _get_video(db, video_id)
    require_video(db, video, principal, need="editor")
    jobs = []
    if payload.engine in ("whisper", "both"):
        jobs.append(
            queue.enqueue(
                db,
                "extract_audio",
                video_id=video.id,
                payload={"chain": ["transcribe"]},
                priority=80,
            )
        )
    if payload.engine in ("gemini", "both"):
        chain = ["merge"] if payload.engine == "both" else []
        jobs.append(
            queue.enqueue(
                db,
                "gemini_transcribe",
                video_id=video.id,
                payload={"chain": chain},
                priority=80,
            )
        )
    db.commit()
    return GenerateResponse(jobs=[JobOut.model_validate(j) for j in jobs])


@router.get("/{video_id}/transcript/segments", response_model=SegmentList)
def get_segments(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> SegmentList:
    require_video(db, _get_video(db, video_id), principal)
    transcript = _current_transcript(db, video_id)
    if transcript is None or not transcript.segments:
        return SegmentList(segments=[])
    return SegmentList(segments=transcript.segments)
