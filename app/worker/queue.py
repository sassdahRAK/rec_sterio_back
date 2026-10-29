"""Postgres-backed job queue: enqueue, atomic claim, retry with backoff.

Claiming uses ``SELECT ... FOR UPDATE SKIP LOCKED`` inside a transaction so
two workers can never run the same job.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Job

logger = logging.getLogger("recsterio.queue")

# A job that stayed 'running' longer than this is considered abandoned and can
# be reclaimed (worker crash / restart safety).
STALE_RUNNING_SECONDS = 1800

# `jobs.status` has no dedicated terminal state for an operator cancellation
# (the frozen contract lists queued/running/succeeded/failed/dead), so a
# cancelled job is `failed` with this marker in `last_error`. Every other
# writer treats that pair as terminal: once it is set, a handler that is still
# running may no longer flip the row to succeeded/failed/dead.
CANCELLED_MARKER = "cancelled"

# Statuses a job can still legitimately transition out of.
ACTIVE_STATUSES = ("queued", "running")


class JobCancelled(Exception):
    """Raised when a running handler notices the operator cancelled its job."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_not_cancelled(db: Session, job: Job) -> None:
    """Checkpoint for long handlers: stop before writing more work.

    Reads the job row straight from the database (a column query bypasses the
    session identity map) so a cancel committed by another session is visible
    even though this handler's in-memory ``job`` still says ``running``.

    A legacy ``status = 'cancelled'`` row (a value no committed code writes; the
    boot batch rewrites it into the canonical pair) counts as cancelled too, so
    a handler still unwinding cannot write past it before that migration runs.
    """
    row = db.execute(
        select(Job.status, Job.last_error).where(Job.id == job.id)
    ).one_or_none()
    if row is None:
        return
    legacy_cancelled = row[0] == "cancelled"
    canonical_cancelled = row[0] == "failed" and row[1] == CANCELLED_MARKER
    if legacy_cancelled or canonical_cancelled:
        raise JobCancelled(f"job {job.id} was cancelled")


def enqueue(
    db: Session,
    kind: str,
    video_id: UUID | None = None,
    payload: Optional[Dict[str, Any]] = None,
    priority: int = 100,
    max_attempts: Optional[int] = None,
    scheduled_at: Optional[datetime] = None,
) -> Job:
    job = Job(
        kind=kind,
        video_id=video_id,
        status="queued",
        payload=payload or {},
        attempts=0,
        max_attempts=max_attempts if max_attempts is not None else settings.worker_max_attempts,
        priority=priority,
        scheduled_at=scheduled_at or utcnow(),
    )
    db.add(job)
    db.flush()
    return job


def claim_job(db: Session) -> Optional[Job]:
    """Atomically claim the next eligible job. Returns None if there is nothing to do."""
    now = utcnow()

    # Reclaim jobs abandoned by a crashed worker.
    stale_before = now - timedelta(seconds=STALE_RUNNING_SECONDS)
    db.execute(
        update(Job)
        .where(Job.status == "running", Job.started_at.isnot(None), Job.started_at < stale_before)
        .values(status="queued", scheduled_at=now, last_error="reclaimed after stale run")
    )

    stmt = (
        select(Job)
        .where(
            Job.status == "queued",
            Job.scheduled_at <= now,
        )
        .order_by(Job.priority.asc(), Job.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = db.execute(stmt).scalar_one_or_none()
    if job is None:
        db.commit()
        return None

    job.status = "running"
    job.attempts = (job.attempts or 0) + 1
    job.started_at = now
    job.finished_at = None
    db.commit()
    db.refresh(job)
    return job


def backoff_seconds(attempts: int) -> int:
    """Exponential backoff: 2**attempts minutes (2, 4, 8, ...)."""
    return int(60 * (2 ** max(0, attempts)))


def succeed(db: Session, job: Job) -> None:
    """Mark the job succeeded, unless it was cancelled while the handler ran.

    The transition is a conditional UPDATE on the database row, not an
    assignment on the (possibly stale) in-memory object: a cancel committed by
    another session during a long handler wins, and the handler's completion
    can no longer overwrite it.
    """
    db.execute(
        update(Job)
        .where(Job.id == job.id, Job.status.in_(ACTIVE_STATUSES))
        .values(status="succeeded", finished_at=utcnow(), last_error=None)
        .execution_options(synchronize_session=False)
    )
    db.commit()
    db.refresh(job)


def fail(db: Session, job: Job, error: str, permanent: bool = False) -> Job:
    """Record a failure; requeue with exponential backoff or mark the job dead."""
    values: Dict[str, Any] = {
        "last_error": error[:4000],
        "finished_at": utcnow(),
    }
    if permanent:
        # Not retryable on its own (protected content, missing credentials...).
        values["status"] = "failed"
    elif job.attempts < job.max_attempts:
        values["status"] = "queued"
        values["scheduled_at"] = utcnow() + timedelta(seconds=backoff_seconds(job.attempts))
    else:
        values["status"] = "dead"
    # Guarded like succeed(): a cancelled job is already terminal and must not
    # be requeued, marked dead or have its error overwritten.
    db.execute(
        update(Job)
        .where(Job.id == job.id, Job.status.in_(ACTIVE_STATUSES))
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    db.commit()
    db.refresh(job)
    return job


def defer(
    db: Session, job: Job, reason: str, delay_seconds: float = 30.0
) -> Job:
    """Re-queue an active job without counting the round as a failure.

    A handler that is waiting for another job (a merge waiting for the Whisper
    or Gemini transcript) returns control this way: the job stays claimable
    after ``delay_seconds`` and its terminal-state bookkeeping is untouched.
    """
    db.execute(
        update(Job)
        .where(Job.id == job.id, Job.status.in_(ACTIVE_STATUSES))
        .values(
            status="queued",
            scheduled_at=utcnow() + timedelta(seconds=delay_seconds),
            last_error=reason[:4000],
            finished_at=None,
        )
        .execution_options(synchronize_session=False)
    )
    db.commit()
    db.refresh(job)
    return job


def cancel(db: Session, job: Job) -> Job:
    """Cancel a job; the terminal state survives whatever the handler does next."""
    db.execute(
        update(Job)
        .where(Job.id == job.id, Job.status.in_(ACTIVE_STATUSES))
        .values(
            status="failed",
            last_error=CANCELLED_MARKER,
            finished_at=utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    # A cancelled merge must not leave its `merges` row stuck at 'running'
    # forever — there would be no way to recover it from the dashboard.
    if job.kind == "merge" and job.video_id:
        from app.models import Merge

        stmt = update(Merge).where(
            Merge.video_id == job.video_id,
            Merge.status.in_(("pending", "running")),
        )
        merge_id = (job.payload or {}).get("merge_id")
        if merge_id:
            try:
                stmt = stmt.where(Merge.id == UUID(str(merge_id)))
            except (TypeError, ValueError):
                pass
        db.execute(
            stmt.values(status="failed", error=CANCELLED_MARKER).execution_options(
                synchronize_session=False
            )
        )
    db.commit()
    db.refresh(job)
    return job


def requeue(db: Session, job: Job) -> Job:
    """Reset a failed/dead job back to queued (attempts = 0)."""
    job.status = "queued"
    job.attempts = 0
    job.last_error = None
    job.scheduled_at = utcnow()
    job.started_at = None
    job.finished_at = None
    db.commit()
    return job
