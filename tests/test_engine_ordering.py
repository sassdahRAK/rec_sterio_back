"""Merge ordering and deferral (F4 + F7).

* `engine=both` starts Whisper and Gemini in parallel. If Gemini's job finishes
  first, the chained merge must wait for the Whisper transcript (and queue the
  Whisper pass if it is missing) instead of merging against nothing.
* A merge with no Gemini transcript must enqueue a Gemini pass and defer, not
  fail with "No Gemini transcript to merge".
* Deferring must not storm the queue: a dependency pass that already failed
  permanently is not enqueued again, and the merge says on its own row why it
  is waiting (then fails with that reason instead of burning 120 rounds).
"""
from __future__ import annotations

import wave

from app.db import SessionLocal
from app.models import Job, Merge, Transcript
from app.services import asr as asr_service
from app.services import gemini as gemini_service
from app.services import storage
from app.worker import queue as job_queue
from app.worker.runner import run_job

URL = "https://www.youtube.com/watch?v=recsterio-engine-order-001"

WHISPER_SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "the birch can use lid on the smooth planks", "language": "en"},
    {"start": 4.0, "end": 7.0, "text": "glue the sheet to the dark blue background", "language": "en"},
]

GEMINI_SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "the birch canoe slid on the smooth planks", "language": "en"},
    {"start": 4.0, "end": 7.0, "text": "glue the sheet to the dark blue background", "language": "en"},
]


class FakeGemini:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe_audio(self, audio_path, model=None):
        return [
            asr_service.ASRSegment(s["start"], s["end"], s["text"], s["language"])
            for s in GEMINI_SEGMENTS
        ]


class FakeWhisper:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio_path):
        return [
            asr_service.ASRSegment(s["start"], s["end"], s["text"], s["language"])
            for s in WHISPER_SEGMENTS
        ]


def _create_video(client, headers) -> str:
    resp = client.post("/api/v1/videos", json={"source_url": URL}, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _write_media_and_audio(video_id: str) -> None:
    storage.video_dir(video_id)
    (storage.video_dir(video_id) / "media.mp4").write_bytes(b"not-a-real-video")
    audio = storage.audio_file(video_id)
    with wave.open(str(audio), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)


def _fake_mp3(media_path, out_path, timeout_seconds=None):
    out_path.write_bytes(b"ID3" + b"\x00" * 300)
    return out_path


def _add_transcript(video_id: str, version: int, source: str, segments: list[dict]) -> str:
    with SessionLocal() as db:
        row = Transcript(
            video_id=video_id,
            version=version,
            source=source,
            language="en",
            content=" ".join(s["text"] for s in segments),
            segments=segments,
            status="ready",
            created_by=source,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return str(row.id)


def _job_of_kind(kind: str, statuses: tuple[str, ...] = ("queued", "running")):
    with SessionLocal() as db:
        job = (
            db.query(Job)
            .filter(Job.kind == kind, Job.status.in_(statuses))
            .order_by(Job.created_at.asc())
            .first()
        )
        return str(job.id) if job is not None else None


def _run_job(job_id: str) -> None:
    with SessionLocal() as db:
        run_job(db, db.get(Job, job_id))


def _patch_engines(monkeypatch) -> None:
    monkeypatch.setattr(gemini_service, "GeminiClient", FakeGemini)
    monkeypatch.setattr(
        asr_service,
        "transcribe_audio",
        lambda path, engine=None, engine_name=None: FakeWhisper().transcribe(path),
    )
    monkeypatch.setattr(asr_service, "extract_mp3", _fake_mp3)


def test_missing_gemini_transcript_enqueues_a_pass_and_defers(client, auth_headers, monkeypatch):
    _patch_engines(monkeypatch)
    video_id = _create_video(client, auth_headers)
    _write_media_and_audio(video_id)
    _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)

    created = client.post(
        f"/api/v1/videos/{video_id}/transcript/merge", json={}, headers=auth_headers
    )
    assert created.status_code == 202, created.text
    merge_id = created.json()["id"]
    assert created.json()["status"] == "pending"

    gemini_job = _job_of_kind("gemini_transcribe")
    assert gemini_job is not None, "the API must enqueue the missing Gemini pass"
    merge_job = _job_of_kind("merge")
    assert merge_job is not None

    # Running the merge first must defer, not fail.
    _run_job(merge_job)
    with SessionLocal() as db:
        job = db.get(Job, merge_job)
        assert job.status == "queued", job.last_error
        assert "Gemini" in (job.last_error or "")
        row = db.get(Merge, merge_id)
        assert row.status == "pending", row.error
        assert row.diff is None
        # ...and the waiting reason is on the row, not only on the job.
        assert row.error and "Gemini" in row.error, row.error
        assert "\n" not in row.error and len(row.error) <= 400

    # The Gemini pass lands; the merge then completes on its own.
    _run_job(gemini_job)
    _run_job(merge_job)

    with SessionLocal() as db:
        row = db.get(Merge, merge_id)
        assert row.status == "ready", row.error
        assert row.error is None, "a ready merge must not keep its waiting reason"
        gemini_rows = (
            db.query(Transcript)
            .filter(Transcript.video_id == video_id, Transcript.source == "gemini")
            .all()
        )
        assert len(gemini_rows) == 1
        assert str(row.gemini_transcript_id) == str(gemini_rows[0].id)


def test_engine_both_waits_for_the_whisper_transcript(client, auth_headers, monkeypatch):
    _patch_engines(monkeypatch)
    video_id = _create_video(client, auth_headers)
    _write_media_and_audio(video_id)

    generated = client.post(
        f"/api/v1/videos/{video_id}/transcript/generate",
        json={"engine": "both"},
        headers=auth_headers,
    )
    assert generated.status_code == 202, generated.text
    kinds = sorted(job["kind"] for job in generated.json()["jobs"])
    assert kinds == ["extract_audio", "gemini_transcribe"]

    # Gemini finishes first and chains a merge against a video with no base
    # transcript yet.
    _run_job(_job_of_kind("gemini_transcribe"))
    merge_job = _job_of_kind("merge")
    assert merge_job is not None, "the Gemini job must chain the merge"
    _run_job(merge_job)

    with SessionLocal() as db:
        merge_row = db.query(Merge).filter(Merge.video_id == video_id).one()
        assert merge_row.status == "pending", merge_row.error
        assert merge_row.diff is None
        assert (
            db.query(Transcript)
            .filter(Transcript.video_id == video_id, Transcript.source == "merged")
            .count()
            == 0
        )
        deferred = db.get(Job, merge_job)
        assert deferred.status == "queued", deferred.last_error
        # The Whisper pass was already queued by /generate; the handler must not
        # duplicate it.
        extract_jobs = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "extract_audio")
            .all()
        )
        assert len(extract_jobs) == 1

    # Whisper runs; only then may the merge succeed.
    _run_job(_job_of_kind("extract_audio"))
    _run_job(_job_of_kind("transcribe"))
    _run_job(merge_job)

    with SessionLocal() as db:
        merge_row = db.query(Merge).filter(Merge.video_id == video_id).one()
        assert merge_row.status == "ready", merge_row.error
        base = db.get(Transcript, merge_row.base_transcript_id)
        assert base.source == "whisper"
        assert merge_row.diff["stats"]["base_segments"] == len(WHISPER_SEGMENTS)


def test_engine_both_enqueues_whisper_when_no_pass_is_in_flight(client, auth_headers, monkeypatch):
    """If the Whisper job vanished, the merge queues one instead of failing."""
    _patch_engines(monkeypatch)
    video_id = _create_video(client, auth_headers)
    _write_media_and_audio(video_id)

    # Gemini transcript exists, but nothing else: no base, no queued Whisper.
    _add_transcript(video_id, 1, "gemini", GEMINI_SEGMENTS)
    with SessionLocal() as db:
        job = job_queue.enqueue(db, "merge", video_id=video_id, payload={"chain": []})
        db.commit()
        merge_job = str(job.id)

    _run_job(merge_job)

    with SessionLocal() as db:
        extract_job = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "extract_audio")
            .one_or_none()
        )
        assert extract_job is not None, "the merge must enqueue the Whisper pass"
        assert (
            db.query(Transcript)
            .filter(Transcript.video_id == video_id, Transcript.source == "merged")
            .count()
            == 0
        )
        assert db.get(Job, merge_job).status == "queued"


def test_a_failed_gemini_pass_is_not_reenqueued_and_the_merge_reports_it(
    client, auth_headers
):
    """The queue-storm regression: one stuck merge used to spawn ~120 failed
    Gemini jobs, and the merge row stayed pending with no explanation.

    This environment has no GEMINI_API_KEY, so the pass fails permanently on
    its first run. The merge must then (a) not enqueue another copy on the next
    round, (b) say on its own row what is missing, one bounded line, and
    (c) fail instead of deferring for an hour.
    """
    video_id = _create_video(client, auth_headers)
    _write_media_and_audio(video_id)
    _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)

    created = client.post(
        f"/api/v1/videos/{video_id}/transcript/merge", json={}, headers=auth_headers
    )
    assert created.status_code == 202, created.text
    merge_id = created.json()["id"]
    gemini_job = _job_of_kind("gemini_transcribe")
    merge_job = _job_of_kind("merge")
    assert gemini_job is not None and merge_job is not None

    # The Gemini pass fails permanently (no API key).
    _run_job(gemini_job)
    with SessionLocal() as db:
        failed = db.get(Job, gemini_job)
        assert failed.status == "failed", failed.last_error
        assert "GEMINI_API_KEY" in (failed.last_error or "")

    # First deferral round: it must give up immediately, not defer.
    _run_job(merge_job)
    # A second round (as a stale worker would run it) must not re-enqueue.
    _run_job(merge_job)

    with SessionLocal() as db:
        gemini_jobs = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "gemini_transcribe")
            .all()
        )
        assert len(gemini_jobs) == 1, "a terminally failed pass must not be re-enqueued"

        row = db.get(Merge, merge_id)
        assert row.status == "failed", row.status
        assert row.error, "a failed merge must record why it could not proceed"
        assert "Gemini" in row.error and "GEMINI_API_KEY" in row.error, row.error
        assert "\n" not in row.error, "Merge.error must stay one line"
        assert len(row.error) <= 400, len(row.error)

        job = db.get(Job, merge_job)
        assert job.status == "failed", job.status
        assert "_defers" not in (job.payload or {}), "it must fail, not defer"
