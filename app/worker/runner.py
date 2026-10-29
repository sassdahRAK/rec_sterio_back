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

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Job, Merge, Transcript, Video
from app.schemas import sanitize_technical_error
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


class JobDeferred(Exception):
    """The handler is waiting for another job; re-queue without burning a try.

    A merge that has no Whisper/Gemini transcript yet must not fail: it stays
    pending and this job is scheduled to run again shortly.
    """

    def __init__(self, reason: str, delay_seconds: float = 30.0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.delay_seconds = delay_seconds


# A merge may poll for its input transcripts for a bounded time (30s * 120 =
# one hour at the default delay) before it is treated as a real failure.
MAX_MERGE_DEFERS = 120
MERGE_DEFER_SECONDS = 30.0

# A waiting merge records *why* on its own row (`Merge.error` is surfaced by
# MergeOut), otherwise the console shows an unexplained pending merge. One
# line, bounded, path-free — the same rule the API applies to step errors.
MERGE_WAIT_ERROR_LIMIT = 300

_WHISPER_KINDS = ("extract_audio", "transcribe")

# Payload keys a chained job must carry forward: the chain only records the
# next `kind`, but the merge job needs to know which `merges` row it belongs to.
_CHAIN_CONTEXT_KEYS = ("merge_id", "base_transcript_id", "gemini_transcript_id", "model")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def lock_video(db: Session, video_id: UUID) -> None:
    """Take the video's row lock for the rest of the transaction.

    Transcript versions are allocated as ``max(version) + 1``; there is no
    unique constraint on (video_id, version), so two concurrent writers could
    otherwise pick the same number. Serialising on the parent video row makes
    the read-then-insert atomic without a schema change.
    """
    db.execute(select(Video.id).where(Video.id == video_id).with_for_update())


def next_version(db: Session, video_id: UUID) -> int:
    current = db.execute(
        select(Transcript.version)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    return (current or 0) + 1


def allocate_version(db: Session, video_id: UUID) -> int:
    """Next transcript version, safe against concurrent writers for the video."""
    lock_video(db, video_id)
    return next_version(db, video_id)


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
        version=allocate_version(db, video_id),
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
    # Carry the merge context forward, otherwise the chained merge job would
    # create a second `merges` row and leave the original one pending forever.
    for key in _CHAIN_CONTEXT_KEYS:
        if key in (job.payload or {}):
            payload[key] = job.payload[key]
    video_id = job.video_id
    queue.enqueue(db, nxt, video_id=video_id, payload=payload, priority=job.priority)


def _set_video(db: Session, video: Video, **values: Any) -> None:
    for key, value in values.items():
        setattr(video, key, value)
    db.flush()


def _cleanup_cloud_temp(video: Video, media: Path) -> None:
    """Delete the temp local copy of a cloud-stored video after extraction."""
    if video.storage_backend and media.name.endswith(".cloud-tmp"):
        try:
            media.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove cloud temp %s", media)
        if video.local_path == str(media):
            video.local_path = None


def _ensure_audio(db: Session, video: Video) -> Path:
    """Make sure a 16 kHz mono wav exists for the video (idempotent)."""
    media = storage.find_media_file(video.id)
    if media is None:
        raise PermanentJobError(
            "No downloaded media found for this video; the download job must run first."
        )
    audio = storage.audio_file(video.id)
    if audio.exists() and audio.stat().st_size > 44:
        _cleanup_cloud_temp(video, media)
        return audio
    try:
        asr_service.extract_audio(media, audio)
    except RuntimeError as exc:
        _cleanup_cloud_temp(video, media)
        raise PermanentJobError(f"Audio extraction failed: {exc}") from exc
    # Best-effort mp3 BEFORE we clean up the cloud temp so the source file
    # is still available for extraction.
    _ensure_mp3(db, video, media=media)
    _cleanup_cloud_temp(video, media)
    _set_video(db, video, status="transcribing")
    return audio


def _ensure_mp3(db: Session, video: Video, media: Optional[Path] = None) -> None:
    """Make sure an .mp3 listening copy exists for the video (best effort).

    Unlike the wav, the mp3 is a convenience for humans, never an input the
    pipeline needs: a missing media file or a failed extraction is only
    logged, because killing the chain over it would cost ASR the wav.

    Pass `media` directly to bypass `find_media_file` — useful when the caller
    already holds the path and the cloud-tmp has not yet been deleted.
    """
    if media is None:
        media = storage.find_media_file(video.id)
    if media is None:
        return
    mp3 = storage.audio_mp3_file(video.id)
    if mp3.exists() and mp3.stat().st_size > 200:
        return
    try:
        asr_service.extract_mp3(media, mp3)
    except Exception as exc:  # noqa: BLE001 - non-fatal by design
        logger.warning("mp3 extraction failed for %s: %s", video.id, exc)


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------


def handle_download(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")

    queue.ensure_not_cancelled(db, job)

    # Idempotency: the file may already be on disk from an earlier run.
    existing = storage.find_media_file(video.id)
    if existing and video.checksum_sha256:
        current_sha = storage.sha256_file(existing)
        if current_sha == video.checksum_sha256:
            queue.ensure_not_cancelled(db, job)
            _set_video(db, video, status="downloaded", progress=100, local_path=str(existing))
            _continue_chain(db, job)
            return

    _set_video(db, video, status="downloading", progress=0, error=None)
    db.commit()

    # A video ingested by a Telegram user ("telegram:<id>") downloads under
    # that user's own cookies when they uploaded some; every other actor
    # ("system", "admin:<email>", the bot's "telegram:bot") uses server
    # defaults.
    cookiefile = None
    if video.created_by and video.created_by.startswith("telegram:"):
        uid = video.created_by.split(":", 1)[1]
        if uid.isdigit():
            user_cookies = storage.cookie_file(int(uid))
            if user_cookies.exists():
                cookiefile = str(user_cookies)

    def _progress(percent: int) -> None:
        with SessionLocal() as pdb:
            row = pdb.get(Video, video.id)
            if row is not None:
                row.progress = percent
                pdb.commit()

    try:
        result = downloader.download(
            video.source_url, video.id, on_progress=_progress, cookiefile=cookiefile
        )
    except downloader.DownloadError as exc:
        raise PermanentJobError(str(exc)) from exc

    # A download cancelled mid-flight must not mark the video downloaded, write
    # the media metadata or chain the audio/transcription jobs.
    queue.ensure_not_cancelled(db, job)

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
        # TvNa schema (instruction.txt #6). Never overwrite a value a collector
        # has already corrected — the download is a source of defaults, not of
        # truth.
        author_handle=video.author_handle or result.author_handle,
        caption=video.caption or result.caption,
        date_posted=video.date_posted or result.date_posted,
        error=None,
    )
    _continue_chain(db, job)


def handle_extract_audio(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")
    queue.ensure_not_cancelled(db, job)
    audio = _ensure_audio(db, video)
    # Best-effort listening copy; failures inside are logged, never fatal.
    _ensure_mp3(db, video)
    # Cancelled during extraction: no status change and no chained job.
    queue.ensure_not_cancelled(db, job)
    # Only move to "transcribing" when a transcription job actually follows.
    # An extract-only request (upload with start_transcription=false) has an
    # empty chain, and setting "transcribing" would strand the video there
    # forever with no job left to advance it.
    if "transcribe" in _chain(job.payload or {}):
        _set_video(db, video, status="transcribing")
    else:
        _set_video(db, video, status="downloaded", progress=100)
    logger.info("audio extracted: %s", audio)
    _continue_chain(db, job)


def handle_transcribe(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")
    queue.ensure_not_cancelled(db, job)
    audio = _ensure_audio(db, video)

    try:
        segments = asr_service.transcribe_audio(audio)
    except asr_service.AsrUnavailable as exc:
        raise PermanentJobError(str(exc)) from exc

    seg_rows = [s.to_dict() for s in segments]
    content = " ".join(s.text.strip() for s in segments if s.text.strip())
    language = _overall_language(seg_rows)

    # Last checkpoint before the irreversible write: a job cancelled while the
    # ASR ran must not create a transcript version, advance video.status or
    # chain the next job.
    queue.ensure_not_cancelled(db, job)

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
    queue.ensure_not_cancelled(db, job)
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

    # Same rule as the Whisper handler: no write, status change or chained job
    # for a job cancelled while Gemini ran.
    queue.ensure_not_cancelled(db, job)

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


def _latest_ready_transcript(
    db: Session, video_id: UUID, source: Optional[str] = None, exclude_source: Optional[str] = None
) -> Optional[Transcript]:
    """Newest ready transcript, optionally filtered to / away from one source."""
    stmt = select(Transcript).where(
        Transcript.video_id == video_id, Transcript.status == "ready"
    )
    if source:
        stmt = stmt.where(Transcript.source == source)
    if exclude_source:
        stmt = stmt.where(Transcript.source != exclude_source)
    return db.execute(stmt.order_by(Transcript.version.desc()).limit(1)).scalars().first()


def _has_active_job(db: Session, video_id: UUID, kinds: Sequence[str]) -> bool:
    """Is work of one of these kinds already queued/running for the video?"""
    row = db.execute(
        select(Job.id)
        .where(
            Job.video_id == video_id,
            Job.kind.in_(tuple(kinds)),
            Job.status.in_(("queued", "running")),
        )
        .limit(1)
    ).first()
    return row is not None


def _last_finished_job(db: Session, video_id: UUID, kinds: Sequence[str]):
    """Newest finished job of these kinds (status, last_error), or None.

    Ordered by when it finished, not when it was created: a job that was
    retried later must not be shadowed by an older attempt.
    """
    return db.execute(
        select(Job.status, Job.last_error)
        .where(
            Job.video_id == video_id,
            Job.kind.in_(tuple(kinds)),
            Job.status.in_(("succeeded", "failed", "dead")),
        )
        .order_by(
            func.coalesce(Job.finished_at, Job.created_at).desc(),
            Job.created_at.desc(),
        )
        .limit(1)
    ).first()


def _dependency_state(
    db: Session, video_id: UUID, kinds: Sequence[str]
) -> tuple[str, Optional[str]]:
    """Where the pass a merge needs stands: active / failed / succeeded / missing.

    ``failed`` is terminal: nothing will run that pass again on its own, so the
    merge must not enqueue a fresh copy every defer round (one stuck merge used
    to spawn a failed job per round). A still-queued pass that is retrying an
    earlier failure reports that failure as its ``last_error`` so the merge can
    say why it is waiting.
    """
    active = _has_active_job(db, video_id, kinds)
    row = _last_finished_job(db, video_id, kinds)
    if active:
        retrying = row is not None and row[0] in ("failed", "dead")
        return "active", row[1] if retrying else None
    if row is None:
        return "missing", None
    if row[0] in ("failed", "dead"):
        return "failed", row[1]
    return "succeeded", None


def _merge_wait_message(piece: str, last_error: Optional[str] = None) -> str:
    """One-line `Merge.error` for a merge that is waiting on `piece`."""
    detail = sanitize_technical_error(last_error, MERGE_WAIT_ERROR_LIMIT)
    message = f"Waiting for {piece}"
    if detail:
        message = f"{message}, which last failed: {detail}"
    return sanitize_technical_error(message, MERGE_WAIT_ERROR_LIMIT) or message


def _merge_dependency_failed(pass_name: str, last_error: Optional[str]) -> str:
    """Permanent reason: the pass this merge needs has already failed."""
    detail = sanitize_technical_error(last_error, MERGE_WAIT_ERROR_LIMIT) or "no further detail"
    return (
        sanitize_technical_error(
            f"Merge failed: the {pass_name} transcript pass failed and will not be retried: {detail}",
            MERGE_WAIT_ERROR_LIMIT,
        )
        or f"Merge failed: the {pass_name} transcript pass failed."
    )


def _defer_merge(
    db: Session,
    job: Job,
    merge_row: Merge,
    reason: str,
    last_error: Optional[str] = None,
) -> None:
    """Leave the merge pending and ask the queue to run this job again later."""
    payload = dict(job.payload or {})
    defers = int(payload.get("_defers") or 0) + 1
    if defers > MAX_MERGE_DEFERS:
        raise PermanentJobError(
            f"Merge gave up after waiting for {reason}. Check the worker logs."
        )
    payload["_defers"] = defers
    job.payload = payload
    # Contract statuses are pending|running|ready|failed: a merge waiting for
    # its inputs is back to `pending` (it has not run successfully yet). The
    # reason travels with the row so the console does not show a bare pending.
    merge_row.status = "pending"
    merge_row.error = _merge_wait_message(reason, last_error)
    db.flush()
    raise JobDeferred(reason, MERGE_DEFER_SECONDS)


def handle_merge(db: Session, job: Job) -> None:
    video = db.get(Video, job.video_id) if job.video_id else None
    if video is None:
        raise PermanentJobError("Video no longer exists.")

    # Cheap checkpoint first: the row may have been cancelled between claim and
    # handler start (a cancel sets job.status=failed + last_error=cancelled).
    queue.ensure_not_cancelled(db, job)

    payload = job.payload or {}
    merge_row: Optional[Merge] = None
    if payload.get("merge_id"):
        merge_row = db.get(Merge, UUID(str(payload["merge_id"])))
    if merge_row is None:
        merge_row = Merge(video_id=video.id, status="running", model=settings.gemini_model)
        db.add(merge_row)
        db.flush()
        # Remember the row on the job: a chained merge (`engine=both`) arrives
        # without a merge_id, and without this every deferred re-run would
        # create another `merges` row and strand the previous one at pending.
        payload = dict(payload)
        payload["merge_id"] = str(merge_row.id)
        job.payload = payload
    else:
        # Re-read the row: another session may have cancelled this merge while
        # the job sat in the queue. A cancelled merge is terminal.
        db.refresh(merge_row)
        if merge_row.status == "failed" and merge_row.error == queue.CANCELLED_MARKER:
            raise queue.JobCancelled(f"merge {merge_row.id} was cancelled")

    merge_row.status = "running"
    merge_row.error = None
    db.flush()

    base = None
    if payload.get("base_transcript_id"):
        base = db.get(Transcript, UUID(str(payload["base_transcript_id"])))
        if base is not None and (base.video_id != video.id or base.status != "ready"):
            base = None
    if base is None:
        base = _latest_ready_transcript(db, video.id, source="whisper")
    if base is None:
        base = _latest_ready_transcript(db, video.id, exclude_source="gemini")
    if base is None:
        # `engine=both` starts Whisper and Gemini in parallel; Gemini may finish
        # first. Never merge against a missing base: make sure Whisper work is
        # queued and wait for it instead of failing the merge.
        whisper_state, whisper_error = _dependency_state(db, video.id, _WHISPER_KINDS)
        if whisper_state == "failed":
            # Re-enqueueing here is exactly the storm: every round would create
            # another extract/transcribe pair that fails the same way.
            raise PermanentJobError(
                _merge_dependency_failed("Whisper", whisper_error)
            )
        if whisper_state != "active":
            queue.enqueue(
                db,
                "extract_audio",
                video_id=video.id,
                payload={"chain": ["transcribe"]},
                priority=80,
            )
        _defer_merge(
            db, job, merge_row, "the Whisper transcript", whisper_error
        )

    gem = None
    if payload.get("gemini_transcript_id"):
        gem = db.get(Transcript, UUID(str(payload["gemini_transcript_id"])))
        if gem is not None and (gem.video_id != video.id or gem.status != "ready"):
            gem = None
    if gem is None or gem.id == base.id:
        gem = _latest_ready_transcript(db, video.id, source="gemini")
    if gem is not None and gem.id == base.id:
        gem = None
    if gem is None:
        # The contract: "If gemini_transcript_id is omitted, the worker uses the
        # most recent source=gemini transcript, or generates one." Generate it
        # and wait rather than failing the merge.
        gemini_state, gemini_error = _dependency_state(db, video.id, ("gemini_transcribe",))
        if gemini_state == "failed":
            # No API key is the normal case: a permanent Gemini failure used to
            # be re-enqueued every 30 s for an hour (~120 failed jobs).
            raise PermanentJobError(
                _merge_dependency_failed("Gemini", gemini_error)
            )
        if gemini_state != "active":
            queue.enqueue(
                db,
                "gemini_transcribe",
                video_id=video.id,
                payload={"chain": []},
                priority=80,
            )
        _defer_merge(
            db, job, merge_row, "the Gemini transcript", gemini_error
        )

    merge_row.base_transcript_id = base.id
    merge_row.gemini_transcript_id = gem.id
    merge_row.model = payload.get("model") or merge_row.model or settings.gemini_model
    db.flush()

    result: MergeResult = merge_transcripts(
        segments_from_rows(base.segments), segments_from_rows(gem.segments)
    )
    diff = merge_result_to_diff(result)
    # Last checkpoint: an operator may have cancelled while the merge ran. The
    # cancelled job and merge row must not be flipped to succeeded/ready.
    queue.ensure_not_cancelled(db, job)
    merge_row.diff = diff
    merge_row.status = "ready"
    merge_row.error = None
    # The merged transcript itself is created by POST /merges/{id}/apply, which
    # allocates the version at promotion time. Creating one here would make
    # "apply" either a no-op or a duplicate version.
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
    except queue.JobCancelled as exc:
        # The operator cancelled this job mid-flight. `queue.cancel` already
        # wrote the terminal state; discard this handler's partial writes and
        # leave the job cancelled (never succeeded/failed).
        logger.info("job %s (%s) stopped: %s", job.id, job.kind, exc)
        db.rollback()
        return
    except JobDeferred as exc:
        # Waiting on another job (a merge waiting for its transcripts):
        # re-queue without counting the round as a failure.
        logger.info("job %s (%s) deferred: %s", job.id, job.kind, exc.reason)
        queue.defer(db, job, exc.reason, exc.delay_seconds)
        return
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
