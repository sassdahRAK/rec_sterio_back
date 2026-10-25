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


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    job.status = "succeeded"
    job.finished_at = utcnow()
    job.last_error = None
    db.commit()


def fail(db: Session, job: Job, error: str, permanent: bool = False) -> Job:
    """Record a failure; requeue with exponential backoff or mark the job dead."""
    job.last_error = error[:4000]
    job.finished_at = utcnow()
    if permanent:
        # Not retryable on its own (protected content, missing credentials...).
        job.status = "failed"
    elif job.attempts < job.max_attempts:
        job.status = "queued"
        job.scheduled_at = utcnow() + timedelta(seconds=backoff_seconds(job.attempts))
    else:
        job.status = "dead"
    db.commit()
    return job


def cancel(db: Session, job: Job) -> Job:
    job.status = "failed"
    job.last_error = "cancelled"
    job.finished_at = utcnow()
    db.commit()
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
