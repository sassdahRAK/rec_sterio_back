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
from app.models import Job, Merge, Transcript, Video
from app.schemas import MergeCreate, MergeListOut, MergeOut, TranscriptOut
from app.services.merger import merge_transcripts, segments_from_rows
from app.worker import queue
from app.worker.runner import allocate_version

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
    needs_gemini_pass = False
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
    else:
        # No id: the worker resolves the newest ready Gemini transcript. If
        # there is none, the merge is deferred (not failed) until a Gemini pass
        # lands, so start that pass now if one is not already in flight.
        has_gemini = db.execute(
            select(Transcript.id)
            .where(
                Transcript.video_id == video.id,
                Transcript.source == "gemini",
                Transcript.status == "ready",
            )
            .limit(1)
        ).first()
        if has_gemini is None:
            in_flight = db.execute(
                select(Job.id)
                .where(
                    Job.video_id == video.id,
                    Job.kind == "gemini_transcribe",
                    Job.status.in_(("queued", "running")),
                )
                .limit(1)
            ).first()
            needs_gemini_pass = in_flight is None

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

    if needs_gemini_pass:
        # The merge job below stays `pending` and re-schedules itself until
        # this pass produces a transcript (handle_merge defers it), so callers
        # do not have to watch the queue themselves.
        queue.enqueue(
            db,
            "gemini_transcribe",
            video_id=video.id,
            payload={"chain": []},
            priority=80,
        )

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
    """Promote the merge result to a NEW current transcript version.

    Every apply allocates the next version for the video (never reuses the row
    a previous apply produced, never assumes version 1). The allocation and the
    insert happen under the video's row lock, so two concurrent applies cannot
    pick the same version.
    """
    merge = _get_merge(db, merge_id)
    # Applying writes a new transcript version, so it needs the editor role on
    # the merge's video rather than just a valid token.
    require_video(db, _get_video(db, merge.video_id), principal, need="editor")

    # Lock the video, then re-read the merge inside this transaction: whatever
    # the row cached (status, merged_transcript_id) is not trusted, and the
    # version is computed from the transcripts as they are *now*.
    version = allocate_version(db, merge.video_id)
    db.refresh(merge)
    if merge.status != "ready" or not merge.diff:
        raise AppError(
            409,
            "conflict",
            "This merge is not ready to be applied.",
            [{"status": merge.status}],
        )

    segments = merge.diff.get("segments") or []
    content = " ".join(
        str(s.get("text", "")).strip() for s in segments if s.get("text")
    ).strip()
    language = next(
        (s.get("language") for s in segments if s.get("language")), None
    )
    if not content:
        # Degenerate diff: rebuild from the current source transcripts.
        content, segments, language = _content_from_sources(db, merge)

    merged = Transcript(
        video_id=merge.video_id,
        version=version,
        source="merged",
        language=language,
        content=content,
        segments=segments,
        status="ready",
        created_by=principal.actor,
    )
    db.add(merged)
    db.flush()
    merge.merged_transcript_id = merged.id
    db.commit()
    db.refresh(merged)
    return TranscriptOut.model_validate(merged)


def _content_from_sources(
    db: Session, merge: Merge
) -> tuple[str, list, str | None]:
    """Re-read the merge's source transcripts and re-run the reconciliation."""
    base = (
        db.get(Transcript, merge.base_transcript_id, populate_existing=True)
        if merge.base_transcript_id
        else None
    )
    gemini = (
        db.get(Transcript, merge.gemini_transcript_id, populate_existing=True)
        if merge.gemini_transcript_id
        else None
    )
    if base is not None and gemini is not None and base.id != gemini.id:
        result = merge_transcripts(
            segments_from_rows(base.segments), segments_from_rows(gemini.segments)
        )
        return result.content, result.segments, base.language or gemini.language
    if base is not None:
        return base.content, base.segments or [], base.language
    return "", [], None
