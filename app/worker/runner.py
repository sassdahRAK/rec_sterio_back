"""Job handlers, one per `jobs.kind`.

Every handler is idempotent: re-running it with the same inputs must not
create duplicate assets (files, transcript versions).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Job, Merge, Transcript, Video
from app.services import asr as asr_service
from app.services import downloader, gemini as gemini_service, storage
from app.services.merger import (
    MergeResult,
    merge_result_to_diff,
    merge_transcripts,
    segments_from_rows,
)
from app.worker import queue

logger = logging.getLogger("recsterio.runner")


class PermanentJobError(Exception):
    """The job can never succeed as-is (protected content, missing config...)."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def next_version(db: Session, video_id: UUID) -> int:
    current = db.execute(
        select(Transcript.version)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    return (current or 0) + 1


def find_identical_transcript(
    db: Session, video_id: UUID, content: str, source: Optional[str] = None
) -> Optional[Transcript]:
    """Look for an existing *ready* transcript with identical content (idempotency)."""
    stmt = select(Transcript).where(
        Transcript.video_id == video_id,
        Transcript.status == "ready",
        Transcript.content == content,
    )
    if source:
        stmt = stmt.where(Transcript.source == source)
    return db.execute(stmt.limit(1)).scalars().first()


def create_transcript(
    db: Session,
    video_id: UUID,
    source: str,
    content: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    language: Optional[str] = None,
    created_by: str = "system",
    status: str = "ready",
) -> Transcript:
    transcript = Transcript(
        video_id=video_id,
        version=next_version(db, video_id),
        source=source,
        language=language,
        content=content,
        segments=segments,
        status=status,
        created_by=created_by,
    )
    db.add(transcript)
    db.flush()
    return transcript


def _chain(payload: Dict[str, Any]) -> List[str]:
    chain = (payload or {}).get("chain") or []
    return [c for c in chain if isinstance(c, str)]


def _continue_chain(db: Session, job: Job) -> None:
    """Enqueue the next job of a chain (extract_audio -> transcribe -> merge)."""
    chain = _chain(job.payload or {})
    if not chain:
        return
    nxt = chain[0]
    remaining = chain[1:]
    payload: Dict[str, Any] = {"chain": remaining}
    video_id = job.video_id
    queue.enqueue(db, nxt, video_id=video_id, payload=payload, priority=job.priority)


def _set_video(db: Session, video: Video, **values: Any) -> None:
    for key, value in values.items():
        setattr(video, key, value)
    db.flush()


def _ensure_audio(db: Session, video: Video) -> Path:
    """Make sure a 16 kHz mono wav exists for the video (idempotent)."""
    media = storage.find_media_file(video.id)
    if media is None:
        raise PermanentJobError(
            "No downloaded media found for this video; the download job must run first."
        )
    audio = storage.audio_file(video.id)
    if audio.exists() and audio.stat().st_size > 44:
        return audio
    try:
        asr_service.extract_audio(media, audio)
    except RuntimeError as exc:
        raise PermanentJobError(f"Audio extraction failed: {exc}") from exc
    _set_video(db, video, status="transcribing")
    return audio


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------


def handle_download(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")

    # Idempotency: the file may already be on disk from an earlier run.
    existing = storage.find_media_file(video.id)
    if existing and video.checksum_sha256:
        current_sha = storage.sha256_file(existing)
        if current_sha == video.checksum_sha256:
            _set_video(db, video, status="downloaded", progress=100, local_path=str(existing))
            _continue_chain(db, job)
            return

    _set_video(db, video, status="downloading", progress=0, error=None)
    db.commit()

    def _progress(percent: int) -> None:
        with SessionLocal() as pdb:
            row = pdb.get(Video, video.id)
            if row is not None:
                row.progress = percent
                pdb.commit()

    try:
        result = downloader.download(video.source_url, video.id, on_progress=_progress)
    except downloader.DownloadError as exc:
        raise PermanentJobError(str(exc)) from exc

    known = {"youtube", "tiktok", "facebook", "instagram"}
    _set_video(
        db,
        video,
        status="downloaded",
        progress=100,
        local_path=str(result.path),
        file_size=result.file_size,
        duration_seconds=result.duration_seconds,
        mime_type=result.mime_type,
        checksum_sha256=result.sha256,
        title=video.title or result.title,
        thumbnail_url=video.thumbnail_url or result.thumbnail_url,
        # The contract only allows the known platform values; the platform
        # detected from the URL wins unless the extractor knows it better.
        platform=result.platform if result.platform in known else video.platform,
        error=None,
    )
    _continue_chain(db, job)


def handle_extract_audio(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")
    audio = _ensure_audio(db, video)
    _set_video(db, video, status="transcribing")
    logger.info("audio extracted: %s", audio)
    _continue_chain(db, job)


def handle_transcribe(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")
    audio = _ensure_audio(db, video)

    try:
        segments = asr_service.transcribe_audio(audio)
    except asr_service.AsrUnavailable as exc:
        raise PermanentJobError(str(exc)) from exc

    seg_rows = [s.to_dict() for s in segments]
    content = " ".join(s.text.strip() for s in segments if s.text.strip())
    language = _overall_language(seg_rows)

    existing = find_identical_transcript(db, video.id, content, source="whisper")
    if existing is not None:
        logger.info("transcription unchanged for video %s; keeping version %s",
                    video.id, existing.version)
    else:
        create_transcript(
            db,
            video.id,
            source="whisper",
            content=content,
            segments=seg_rows,
            language=language,
            created_by="whisper",
        )
    _set_video(db, video, status="transcribed", error=None)
    _continue_chain(db, job)


def handle_gemini_transcribe(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")
    audio = _ensure_audio(db, video)

    client = gemini_service.GeminiClient()
    try:
        segments = client.transcribe_audio(audio)
    except gemini_service.GeminiError as exc:
        message = str(exc)
        if "GEMINI_API_KEY is not configured" in message:
            raise PermanentJobError(message) from exc
        raise RuntimeError(message) from exc

    seg_rows = [s.to_dict() for s in segments]
    content = " ".join(s.text.strip() for s in segments if s.text.strip())
    language = _overall_language(seg_rows)

    existing = find_identical_transcript(db, video.id, content, source="gemini")
    if existing is None:
        create_transcript(
            db,
            video.id,
            source="gemini",
            content=content,
            segments=seg_rows,
            language=language,
            created_by="gemini",
        )
    _set_video(db, video, status="transcribed", error=None)
    _continue_chain(db, job)


def _overall_language(segments: Sequence[Dict[str, Any]]) -> Optional[str]:
    langs = {s.get("language") for s in segments if s.get("language")}
    if not langs:
        return None
    if len(langs) == 1:
        return next(iter(langs))
    return "km-en"


def _latest_transcript(
    db: Session, video_id: UUID, source: Optional[str] = None
) -> Optional[Transcript]:
    stmt = select(Transcript).where(Transcript.video_id == video_id)
    if source:
        stmt = stmt.where(Transcript.source == source)
    stmt = stmt.order_by(Transcript.version.desc())
    return db.execute(stmt.limit(1)).scalars().first()


def handle_merge(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")

    payload = job.payload or {}
    merge_row: Optional[Merge] = None
    if payload.get("merge_id"):
        merge_row = db.get(Merge, UUID(str(payload["merge_id"])))
    if merge_row is None:
        merge_row = Merge(video_id=video.id, status="running", model=settings.gemini_model)
        db.add(merge_row)
        db.flush()

    merge_row.status = "running"
    merge_row.error = None
    db.flush()

    base = None
    if payload.get("base_transcript_id"):
        base = db.get(Transcript, UUID(str(payload["base_transcript_id"])))
    if base is None:
        base = _latest_transcript(db, video.id, source="whisper") or _latest_transcript(db, video.id)
    if base is None:
        raise PermanentJobError(
            "No base transcript to merge - generate a Whisper transcript first."
        )

    gem = None
    if payload.get("gemini_transcript_id"):
        gem = db.get(Transcript, UUID(str(payload["gemini_transcript_id"])))
    if gem is None:
        gem = _latest_transcript(db, video.id, source="gemini")
    if gem is None:
        raise PermanentJobError(
            "No Gemini transcript to merge - run the Gemini transcription first."
        )

    merge_row.base_transcript_id = base.id
    merge_row.gemini_transcript_id = gem.id
    merge_row.model = payload.get("model") or merge_row.model or settings.gemini_model
    db.flush()

    result: MergeResult = merge_transcripts(
        segments_from_rows(base.segments), segments_from_rows(gem.segments)
    )
    diff = merge_result_to_diff(result)
    merge_row.diff = diff
    merge_row.status = "ready"
    merge_row.error = None

    merged = find_identical_transcript(db, video.id, result.content, source="merged")
    if merged is None:
        merged = create_transcript(
            db,
            video.id,
            source="merged",
            content=result.content,
            segments=result.segments,
            language=base.language or gem.language,
            created_by="merged",
        )
    merge_row.merged_transcript_id = merged.id
    _set_video(db, video, status="transcribed", error=None)
    db.flush()


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------

HANDLERS = {
    "download": handle_download,
    "extract_audio": handle_extract_audio,
    "transcribe": handle_transcribe,
    "gemini_transcribe": handle_gemini_transcribe,
    "merge": handle_merge,
}


def run_job(db: Session, job: Job) -> None:
    """Execute one job, handling success/failure bookkeeping."""
    handler = HANDLERS.get(job.kind)
    if handler is None:
        queue.fail(db, job, f"Unknown job kind '{job.kind}'.", permanent=True)
        return
    try:
        handler(db, job)
    except PermanentJobError as exc:
        logger.warning("job %s (%s) failed permanently: %s", job.id, job.kind, exc)
        _fail_running_merge_rows(db, job, str(exc))
        queue.fail(db, job, str(exc), permanent=True)
        _mark_video_failed(db, job, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("job %s (%s) failed", job.id, job.kind)
        message = f"{type(exc).__name__}: {exc}"
        _fail_running_merge_rows(db, job, message)
        queue.fail(db, job, message)
        if job.status == "dead":
            _mark_video_failed(db, job, message)
        return
    queue.succeed(db, job)


def _fail_running_merge_rows(db: Session, job: Job, message: str) -> None:
    """A failed merge job must not leave its `merges` row stuck at 'running'."""
    if job.kind != "merge" or not job.video_id:
        return
    rows = (
        db.execute(
            select(Merge).where(Merge.video_id == job.video_id, Merge.status == "running")
        )
        .scalars()
        .all()
    )
    for row in rows:
        row.status = "failed"
        row.error = message[:2000]
    if rows:
        db.flush()


def _mark_video_failed(db: Session, job: Job, message: str) -> None:
    if not job.video_id:
        return
    video = db.get(Video, job.video_id)
    if video is not None:
        video.status = "failed"
        video.error = message[:2000]
        db.commit()
