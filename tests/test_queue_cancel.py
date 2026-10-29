"""Cancellation is terminal (F5).

Before the fix, `queue.cancel` wrote `status=failed, last_error=cancelled` on
the ORM object, but the running handler still held a stale copy of the job and
its own `queue.succeed`/`queue.fail` overwrote the row afterwards. These tests
prove a handler that completes (or fails) after the cancel cannot flip the job
to succeeded/failed, and that a cancelled merge row is not left `running`.

Every handler now checkpoints with `queue.ensure_not_cancelled` before its
irreversible write, and `POST /jobs/{id}/retry` refuses races with a running
same-kind job (or a cancelled row).
"""
from __future__ import annotations

import wave
from uuid import uuid4

import pytest
from sqlalchemy import text

from app import db as db_module
from app.db import SessionLocal, engine, init_db
from app.models import Job, Merge, Transcript, Video
from app.services import asr as asr_service
from app.services import storage
from app.worker import queue, runner


def _enqueue_and_claim(kind: str, video_id=None, payload: dict | None = None):
    with SessionLocal() as db:
        job = queue.enqueue(db, kind, video_id=video_id, payload=payload or {})
        db.commit()
        job_id = job.id
    with SessionLocal() as db:
        # Claim the same job so `cancel` sees it as running.
        job = db.get(Job, job_id)
        job.status = "running"
        job.attempts = 1
        job.started_at = queue.utcnow()
        db.commit()
        return str(job_id)


def test_handler_completing_after_cancel_does_not_flip_the_job_to_succeeded(
    monkeypatch,
):
    monkeypatch.setitem(runner.HANDLERS, "fake", lambda db, job: None)
    job_id = _enqueue_and_claim("fake")

    # The operator cancels while the handler is running (separate session, like
    # the /jobs/{id}/cancel route).
    with SessionLocal() as db:
        queue.cancel(db, db.get(Job, job_id))

    # The handler now completes normally.
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        runner.run_job(db, job)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == "failed", job.status
        assert job.last_error == "cancelled"
        assert job.status != "succeeded"


def test_handler_failing_after_cancel_does_not_rewrite_the_cancellation(monkeypatch):
    def _boom(db, job):
        raise RuntimeError("late failure")

    monkeypatch.setitem(runner.HANDLERS, "fake", _boom)
    job_id = _enqueue_and_claim("fake")

    with SessionLocal() as db:
        queue.cancel(db, db.get(Job, job_id))

    with SessionLocal() as db:
        runner.run_job(db, db.get(Job, job_id))

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == "failed"
        assert job.last_error == "cancelled", job.last_error


def test_cancelling_a_running_merge_keeps_both_rows_cancelled(client):
    with SessionLocal() as db:
        video = Video(source_url="https://example.org/cancel-merge-001", platform="direct")
        db.add(video)
        db.flush()
        merge = Merge(video_id=video.id, status="running", model="gemini-2.5-pro")
        db.add(merge)
        db.flush()
        job = queue.enqueue(
            db,
            "merge",
            video_id=video.id,
            payload={"merge_id": str(merge.id)},
        )
        db.commit()
        job_id, merge_id, video_id = str(job.id), str(merge.id), str(video.id)

    # Simulate the worker having claimed it.
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        job.status = "running"
        job.started_at = queue.utcnow()
        db.commit()

    with SessionLocal() as db:
        queue.cancel(db, db.get(Job, job_id))

    # The in-flight handler finishes its run afterwards.
    with SessionLocal() as db:
        runner.run_job(db, db.get(Job, job_id))

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        merge = db.get(Merge, merge_id)
        assert job.status == "failed"
        assert job.last_error == "cancelled"
        assert merge.status == "failed", "a cancelled merge must not stay running/ready"
        assert merge.error == "cancelled"


def test_cancelled_job_is_not_requeued_by_a_later_defer():
    """Waiting for another job must not resurrect a cancelled job either."""
    job_id = _enqueue_and_claim("fake")
    with SessionLocal() as db:
        queue.cancel(db, db.get(Job, job_id))
    with SessionLocal() as db:
        queue.defer(db, db.get(Job, job_id), "waiting for the Whisper transcript")
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == "failed"
        assert job.last_error == "cancelled"


def _video_with_audio() -> str:
    """A video whose media + 16 kHz wav already exist (no ffmpeg involved)."""
    with SessionLocal() as db:
        video = Video(
            source_url=f"https://example.org/cancel-transcribe-{uuid4().hex[:8]}",
            platform="direct",
        )
        db.add(video)
        db.commit()
        video_id = str(video.id)
    storage.video_dir(video_id)
    (storage.video_dir(video_id) / "media.mp4").write_bytes(b"not-a-real-video")
    with wave.open(str(storage.audio_file(video_id)), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return video_id


def test_cancelled_transcribe_creates_no_version_and_keeps_the_video_status(
    monkeypatch,
):
    """A cancel during ASR must not write a transcript or advance video.status."""
    video_id = _video_with_audio()
    with SessionLocal() as db:
        video = db.get(Video, video_id)
        before_status = video.status
        job = queue.enqueue(db, "transcribe", video_id=video.id, payload={"chain": []})
        job.status = "running"
        job.attempts = 1
        job.started_at = queue.utcnow()
        db.commit()
        job_id = str(job.id)

    def _cancel_midway(audio_path):
        # The operator cancels while the ASR engine is still running.
        with SessionLocal() as cdb:
            queue.cancel(cdb, cdb.get(Job, job_id))
        return [asr_service.ASRSegment(0.0, 2.0, "hello world", "en")]

    monkeypatch.setattr(asr_service, "transcribe_audio", _cancel_midway)

    with SessionLocal() as db:
        runner.run_job(db, db.get(Job, job_id))

    with SessionLocal() as db:
        assert (
            db.query(Transcript).filter(Transcript.video_id == video_id).count() == 0
        ), "a cancelled transcription must not create a version"
        assert db.get(Video, video_id).status == before_status
        job = db.get(Job, job_id)
        assert job.status == "failed"
        assert job.last_error == "cancelled"


def test_retry_is_refused_while_the_same_video_and_kind_is_running(
    client, auth_headers
):
    """Retrying must not race a handler that is already running the same step."""
    with SessionLocal() as db:
        video = Video(
            source_url="https://example.org/cancel-retry-001", platform="direct"
        )
        db.add(video)
        db.flush()
        target = queue.enqueue(db, "transcribe", video_id=video.id, payload={"chain": []})
        target.status = "failed"
        target.last_error = "RuntimeError: engine crashed"
        target.finished_at = queue.utcnow()
        running = queue.enqueue(db, "transcribe", video_id=video.id, payload={"chain": []})
        running.status = "running"
        running.attempts = 1
        running.started_at = queue.utcnow()
        db.commit()
        target_id, running_id = str(target.id), str(running.id)

    resp = client.post(f"/api/v1/jobs/{target_id}/retry", headers=auth_headers)
    assert resp.status_code == 409, resp.text
    error = resp.json()["error"]
    assert error["code"] == "conflict"
    assert "still running" in error["message"]
    assert error["details"][0]["job_id"] == running_id

    with SessionLocal() as db:
        job = db.get(Job, target_id)
        assert job.status == "failed", "a refused retry must leave the job untouched"
        assert job.last_error == "RuntimeError: engine crashed"
        assert job.attempts == 0
        assert job.started_at is None


def test_retry_of_a_cancelled_job_is_refused(client, auth_headers):
    """A cancelled row is terminal: requeueing it would let the stale handler
    still unwinding overwrite the fresh run."""
    with SessionLocal() as db:
        video = Video(
            source_url="https://example.org/cancel-retry-002", platform="direct"
        )
        db.add(video)
        db.flush()
        job = queue.enqueue(db, "download", video_id=video.id, payload={})
        job.status = "running"
        job.attempts = 1
        job.started_at = queue.utcnow()
        db.flush()
        queue.cancel(db, job)
        job_id = str(job.id)

    resp = client.post(f"/api/v1/jobs/{job_id}/retry", headers=auth_headers)
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "conflict"
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == "failed"
        assert job.last_error == "cancelled"


def _legacy_cancelled_job(kind: str = "fake", video_id=None) -> str:
    """A pre-convention row: `status='cancelled'` with no marker.

    No committed code writes this value; it predates the `failed` + marker
    convention and is what the version-2 boot batch rewrites.
    """
    with SessionLocal() as db:
        job = queue.enqueue(db, kind, video_id=video_id, payload={})
        job.status = "cancelled"
        job.last_error = None
        job.attempts = 1
        job.started_at = queue.utcnow()
        db.commit()
        return str(job.id)


def test_retry_of_a_legacy_cancelled_row_is_refused(client, auth_headers):
    """`status='cancelled'` predates the marker, but is still terminal."""
    with SessionLocal() as db:
        video = Video(
            source_url="https://example.org/cancel-legacy-retry-001", platform="direct"
        )
        db.add(video)
        db.flush()
        job = queue.enqueue(db, "download", video_id=video.id, payload={})
        job.status = "cancelled"
        job.last_error = None
        job.attempts = 1
        job.started_at = queue.utcnow()
        db.commit()
        job_id = str(job.id)

    resp = client.post(f"/api/v1/jobs/{job_id}/retry", headers=auth_headers)
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "conflict"

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == "cancelled", "a refused retry must leave the job untouched"
        assert job.last_error is None
        assert job.attempts == 1


def test_ensure_not_cancelled_recognises_a_legacy_cancelled_row():
    job_id = _legacy_cancelled_job()
    with SessionLocal() as db:
        with pytest.raises(queue.JobCancelled):
            queue.ensure_not_cancelled(db, db.get(Job, job_id))


def test_succeed_does_not_overwrite_a_legacy_cancelled_row():
    """The late-write hole is closed before the boot batch ever runs."""
    job_id = _legacy_cancelled_job()
    with SessionLocal() as db:
        queue.succeed(db, db.get(Job, job_id))
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        assert job.status == "cancelled", "succeed() must not move a cancelled row"
        assert job.last_error is None


def test_migration_normalises_legacy_cancelled_rows(client, monkeypatch):
    """The version-2 boot batch rewrites the legacy status into the marker pair.

    Exercises the batch the way `tests/test_boot_idempotent.py` does: force the
    "database is genuinely below target" path so `init_db` cannot take the
    adoption shortcut, then let it run.
    """
    with SessionLocal() as db:
        null_error = queue.enqueue(db, "fake")
        null_error.status = "cancelled"
        null_error.last_error = None
        kept_error = queue.enqueue(db, "fake")
        kept_error.status = "cancelled"
        kept_error.last_error = "engine exploded"
        db.commit()
        null_id, kept_id = str(null_error.id), str(kept_error.id)

    monkeypatch.setattr(db_module, "_migration_already_applied", lambda conn: False)
    with engine.begin() as conn:
        conn.execute(text("delete from schema_migrations"))
    init_db()

    with SessionLocal() as db:
        normalised = db.get(Job, null_id)
        assert normalised.status == "failed"
        assert normalised.last_error == "cancelled"
        kept = db.get(Job, kept_id)
        assert kept.status == "failed"
        assert kept.last_error == "engine exploded", "a recorded error must survive"
        assert db_module.SCHEMA_VERSION == 2
