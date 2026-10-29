"""Job routes: list, inspect, retry, cancel."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.access import require_operator
from app.auth import Principal, get_principal
from app.db import get_db
from app.errors import AppError
from app.models import Job
from app.schemas import JobListOut, JobOut
from app.worker import queue

router = APIRouter(prefix="/jobs", tags=["jobs"])

SORTABLE = {"created_at", "-created_at", "scheduled_at", "-scheduled_at"}


def _get_job(db: Session, job_id: UUID) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise AppError(404, "not_found", "Job not found.", [{"id": str(job_id)}])
    return job


@router.get("", response_model=JobListOut)
def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    kind: Optional[str] = None,
    video_id: Optional[UUID] = None,
    sort: str = Query("-created_at"),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> JobListOut:
    require_operator(principal)
    stmt = select(Job)
    count_stmt = select(func.count()).select_from(Job)
    if status:
        stmt = stmt.where(Job.status == status)
        count_stmt = count_stmt.where(Job.status == status)
    if kind:
        stmt = stmt.where(Job.kind == kind)
        count_stmt = count_stmt.where(Job.kind == kind)
    if video_id:
        stmt = stmt.where(Job.video_id == video_id)
        count_stmt = count_stmt.where(Job.video_id == video_id)
    total = db.execute(count_stmt).scalar_one()
    if sort not in SORTABLE:
        raise AppError(
            422,
            "validation_failed",
            "sort must be one of created_at, -created_at, scheduled_at.",
            [{"field": "sort", "issue": "unsupported_value", "allowed": sorted(SORTABLE)}],
        )
    column = {
        "created_at": Job.created_at,
        "-created_at": Job.created_at.desc(),
        "scheduled_at": Job.scheduled_at,
        "-scheduled_at": Job.scheduled_at.desc(),
    }[sort]
    stmt = stmt.order_by(column, Job.id).offset((page - 1) * page_size).limit(page_size)
    items = db.execute(stmt).scalars().all()
    return JobListOut(
        items=[JobOut.model_validate(j) for j in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{job_id}", response_model=JobOut)
def get_job(
    job_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> JobOut:
    require_operator(principal)
    return JobOut.model_validate(_get_job(db, job_id))


@router.post("/{job_id}/retry", response_model=JobOut, status_code=202)
def retry_job(
    job_id: UUID,
    db: Session =Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> JobOut:
    require_operator(principal)
    job = _get_job(db, job_id)
    if job.status in ("succeeded", "running", "queued"):
        raise AppError(
            409,
            "conflict",
            "Only failed or dead jobs can be retried.",
            [{"status": job.status}],
        )
    # A cancellation is terminal (queue.CANCELLED_MARKER): the handler may still
    # be unwinding, and requeueing the row would let its late succeed() win over
    # the fresh run. Start a new pass instead (video retry / generate).
    # A legacy `status = "cancelled"` row (the boot batch normalises it) is the
    # same terminal cancellation and must not be requeued either.
    if job.status == "cancelled" or job.last_error == queue.CANCELLED_MARKER:
        raise AppError(
            409,
            "conflict",
            "This job was cancelled, which is terminal; start a new pass instead of retrying it.",
            [{"job_id": str(job.id), "status": job.status, "reason": queue.CANCELLED_MARKER}],
        )
    # Never race a handler that is already running the same step for the same
    # video: the stale run's succeed() could otherwise overwrite the retry.
    if job.video_id is not None:
        running = db.execute(
            select(Job.id)
            .where(
                Job.video_id == job.video_id,
                Job.kind == job.kind,
                Job.status == "running",
                Job.id != job.id,
            )
            .limit(1)
        ).first()
        if running is not None:
            raise AppError(
                409,
                "conflict",
                f"A {job.kind} job for this video is still running; "
                "retrying now would race it.",
                [{"job_id": str(running[0]), "kind": job.kind, "status": "running"}],
            )
    queue.requeue(db, job)
    db.refresh(job)
    return JobOut.model_validate(job)


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(
    job_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> JobOut:
    require_operator(principal)
    job = _get_job(db, job_id)
    if job.status not in ("running", "queued"):
        raise AppError(
            409,
            "conflict",
            "Only running or queued jobs can be cancelled.",
            [{"status": job.status}],
        )
    queue.cancel(db, job)
    db.refresh(job)
    return JobOut.model_validate(job)
