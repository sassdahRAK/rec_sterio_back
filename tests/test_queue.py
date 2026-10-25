"""Job queue: claiming (atomic), retry with exponential backoff, dead/failed."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db import SessionLocal
from app.models import Job
from app.worker import queue


def _enqueue(kind: str = "download", payload: dict | None = None) -> str:
    with SessionLocal() as db:
        job = queue.enqueue(db, kind, payload=payload or {})
        db.commit()
        return str(job.id)


def test_enqueue_creates_queued_job():
    job_id = _enqueue()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == "queued"
        assert job.attempts == 0
        assert job.max_attempts == 3
        assert job.priority == 100


def test_claim_sets_running_and_increments_attempts():
    job_id = _enqueue()
    with SessionLocal() as db:
        job = queue.claim_job(db)
        assert job is not None
        assert str(job.id) == job_id
        assert job.status == "running"
        assert job.attempts == 1
        assert job.started_at is not None


def test_claim_is_exclusive():
    _enqueue()
    with SessionLocal() as db1:
        first = queue.claim_job(db1)
    assert first is not None
    with SessionLocal() as db2:
        second = queue.claim_job(db2)
    assert second is None


def test_claim_skips_jobs_scheduled_in_the_future():
    with SessionLocal() as db:
        queue.enqueue(
            db,
            "download",
            scheduled_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        db.commit()
    with SessionLocal() as db:
        assert queue.claim_job(db) is None


def test_backoff_is_exponential_in_minutes():
    assert queue.backoff_seconds(1) == 120
    assert queue.backoff_seconds(2) == 240
    assert queue.backoff_seconds(3) == 480


def test_failure_requeues_with_backoff_until_max_attempts():
    job_id = _enqueue()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        job.attempts = 1
        before = datetime.now(timezone.utc)
        queue.fail(db, job, "boom")
        assert job.status == "queued"
        assert job.last_error == "boom"
        delay = (job.scheduled_at - before).total_seconds()
        assert 100 <= delay <= 130  # ~2 minutes


def test_failure_at_max_attempts_marks_job_dead():
    job_id = _enqueue()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        job.attempts = 3
        queue.fail(db, job, "boom")
        assert job.status == "dead"


def test_permanent_failure_is_not_retried():
    job_id = _enqueue()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        job.attempts = 1
        queue.fail(db, job, "protected content", permanent=True)
        assert job.status == "failed"
        assert job.last_error == "protected content"


def test_cancel_and_requeue():
    job_id = _enqueue()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        queue.cancel(db, job)
        assert job.status == "failed"
        assert job.last_error == "cancelled"
        queue.requeue(db, job)
        assert job.status == "queued"
        assert job.attempts == 0
