"""Merge routes: start a merge, list merges, inspect a merge, apply a merge."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.access import require_video
from app.auth import Principal, get_principal
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import Merge, Transcript, Video
from app.schemas import MergeCreate, MergeListOut, MergeOut, TranscriptOut
from app.worker import queue
from app.worker.runner import next_version

router = APIRouter(tags=["merges"])


def _get_video(db: Session, video_id: UUID) -> Video:
    video = db.get(Video, video_id)
    if video is None:
        raise AppError(404, "not_found", "Video not found.", [{"id": str(video_id)}])
    return video


def _get_merge(db: Session, merge_id: UUID) -> Merge:
    merge = db.get(Merge, merge_id)
    if merge is None:
        raise AppError(404, "not_found", "Merge not found.", [{"id": str(merge_id)}])
    return merge


@router.post("/videos/{video_id}/transcript/merge", response_model=MergeOut, status_code=202)
def create_merge(
    video_id: UUID,
    payload: MergeCreate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> MergeOut:
    """Enqueue a Gemini-assisted reconciliation job."""
    video = _get_video(db, video_id)
    # A merge enqueues a job and writes a transcript version, so it needs the
    # editor role on the video's folder — not merely a valid token.
    require_video(db, video, principal, need="editor")

    base: Transcript | None = None
    if payload.base_transcript_id:
        base = db.get(Transcript, payload.base_transcript_id)
        if base is None or base.video_id != video.id:
            raise AppError(
                404,
                "not_found",
                "Base transcript not found for this video.",
                [{"base_transcript_id": str(payload.base_transcript_id)}],
            )
    else:
        # The base must be the primary transcript, never the Gemini second
        # opinion - merging Gemini with itself would produce a no-op diff and
        # drop anything that only the primary transcript heard. Prefer Whisper
        # (the independent ASR pass); fall back to a manual/merged correction.
        base = db.execute(
            select(Transcript)
            .where(
                Transcript.video_id == video.id,
                Transcript.status == "ready",
                Transcript.source == "whisper",
            )
            .order_by(Transcript.version.desc())
            .limit(1)
        ).scalar_one_or_none()
        if base is None:
            base = db.execute(
                select(Transcript)
                .where(
                    Transcript.video_id == video.id,
                    Transcript.status == "ready",
                    Transcript.source != "gemini",
                )
                .order_by(Transcript.version.desc())
                .limit(1)
            ).scalar_one_or_none()
        if base is None:
            raise AppError(
                404,
                "not_found",
                "This video has no primary transcript to merge yet.",
                [{"video_id": str(video.id)}],
            )

    gemini: Transcript | None = None
    if payload.gemini_transcript_id:
        gemini = db.get(Transcript, payload.gemini_transcript_id)
        if gemini is None or gemini.video_id != video.id:
            raise AppError(
                404,
                "not_found",
                "Gemini transcript not found for this video.",
                [{"gemini_transcript_id": str(payload.gemini_transcript_id)}],
            )
        if gemini.id == base.id:
            raise AppError(
                409,
                "conflict",
                "The base and Gemini transcripts must be different versions.",
                [{"transcript_id": str(base.id)}],
            )

    merge = Merge(
        video_id=video.id,
        base_transcript_id=base.id,
        gemini_transcript_id=gemini.id if gemini else None,
        status="pending",
        model=payload.model or settings.gemini_model,
    )
    db.add(merge)
    db.flush()

    job_payload = {
        "merge_id": str(merge.id),
        "base_transcript_id": str(base.id),
        "chain": [],
    }
    if gemini is not None:
        job_payload["gemini_transcript_id"] = str(gemini.id)
    if payload.model:
        job_payload["model"] = payload.model

    queue.enqueue(db, "merge", video_id=video.id, payload=job_payload, priority=70)
    db.commit()
    db.refresh(merge)
    return MergeOut.model_validate(merge)


@router.get("/videos/{video_id}/merges", response_model=MergeListOut)
def list_merges(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> MergeListOut:
    require_video(db, _get_video(db, video_id), principal)
    items = db.execute(
        select(Merge).where(Merge.video_id == video_id).order_by(Merge.created_at.desc())
    ).scalars().all()
    return MergeListOut(items=[MergeOut.model_validate(m) for m in items])


@router.get("/merges/{merge_id}", response_model=MergeOut)
def get_merge(
    merge_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> MergeOut:
    merge = _get_merge(db, merge_id)
    require_video(db, _get_video(db, merge.video_id), principal)
    return MergeOut.model_validate(merge)


@router.post("/merges/{merge_id}/apply", response_model=TranscriptOut)
def apply_merge(
    merge_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> TranscriptOut:
    """Promote the merged transcript to be the current transcript."""
    merge = _get_merge(db, merge_id)
    # Applying writes a new transcript version, so it needs the editor role on
    # the merge's video rather than just a valid token.
    require_video(db, _get_video(db, merge.video_id), principal, need="editor")
    if merge.status != "ready" or not merge.diff:
        raise AppError(
            409,
            "conflict",
            "This merge is not ready to be applied.",
            [{"status": merge.status}],
        )
    merged = db.get(Transcript, merge.merged_transcript_id) if merge.merged_transcript_id else None
    if merged is None:
        stats = merge.diff.get("stats") or {}
        segments = merge.diff.get("segments") or []
        content = " ".join(
            str(s.get("text", "")).strip() for s in segments if s.get("text")
        ).strip()
        merged = Transcript(
            video_id=merge.video_id,
            version=next_version(db, merge.video_id),
            source="merged",
            language=None,
            content=content,
            segments=segments,
            status="ready",
            created_by=principal.actor,
        )
        db.add(merged)
        db.flush()
        merge.merged_transcript_id = merged.id
    merged.status = "ready"
    db.commit()
    db.refresh(merged)
    return TranscriptOut.model_validate(merged)
