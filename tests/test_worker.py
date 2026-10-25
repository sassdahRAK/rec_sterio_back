"""Worker behaviour: job handlers, idempotency and graceful degradation."""
from __future__ import annotations

import wave
from pathlib import Path

from app.db import SessionLocal
from app.models import Job, Transcript, Video
from app.services import asr as asr_service
from app.services import gemini as gemini_service
from app.services import storage
from app.worker import queue
from app.worker.runner import handle_gemini_transcribe, handle_transcribe, run_job

URL = "https://www.youtube.com/watch?v=recsterio-worker-001"


def _make_video_with_audio() -> str:
    """Create a video row plus a tiny (silent) wav file on disk."""
    with SessionLocal() as db:
        video = Video(source_url=URL, platform="youtube", status="downloaded", progress=100)
        db.add(video)
        db.commit()
        db.refresh(video)
        video_id = str(video.id)

    storage.video_dir(video_id)
    media = storage.video_dir(video_id) / "media.mp4"
    media.write_bytes(b"not-a-real-video")

    audio = storage.audio_file(video_id)
    with wave.open(str(audio), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    return video_id


class FakeEngine:
    """Pluggable fake ASR engine (no faster-whisper needed)."""

    name = "fake"

    def transcribe(self, audio_path: Path):
        return [
            asr_service.ASRSegment(0.0, 2.0, "hello world", "en"),
            asr_service.ASRSegment(2.0, 4.0, "សូមស្វាគមន៍", "km"),
        ]


def test_transcribe_job_creates_a_transcript(monkeypatch):
    video_id = _make_video_with_audio()
    monkeypatch.setattr(
        asr_service, "transcribe_audio", lambda path, engine=None, engine_name=None: FakeEngine().transcribe(path)
    )
    with SessionLocal() as db:
        job = queue.enqueue(db, "transcribe", video_id=video_id)
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status == "succeeded"
        transcripts = db.query(Transcript).filter(Transcript.video_id == video_id).all()
        assert len(transcripts) == 1
        assert transcripts[0].source == "whisper"
        assert transcripts[0].version == 1
        assert "hello world" in transcripts[0].content
        assert transcripts[0].segments[0]["language"] == "en"
        assert transcripts[0].segments[1]["language"] == "km"
        video = db.get(Video, video_id)
        assert video.status == "transcribed"


def test_transcription_is_idempotent(monkeypatch):
    """Re-running the transcription must not create a spurious new version."""
    video_id = _make_video_with_audio()
    monkeypatch.setattr(
        asr_service, "transcribe_audio", lambda path, engine=None, engine_name=None: FakeEngine().transcribe(path)
    )
    for _ in range(2):
        with SessionLocal() as db:
            job = queue.enqueue(db, "transcribe", video_id=video_id)
            db.commit()
            run_job(db, job)

    with SessionLocal() as db:
        transcripts = db.query(Transcript).filter(Transcript.video_id == video_id).all()
        assert len(transcripts) == 1
        assert transcripts[0].version == 1


def test_missing_asr_engine_fails_the_job_without_crashing(monkeypatch):
    video_id = _make_video_with_audio()

    def _boom(*args, **kwargs):
        raise asr_service.AsrUnavailable("faster-whisper is not installed")

    monkeypatch.setattr(asr_service, "transcribe_audio", _boom)
    with SessionLocal() as db:
        job = queue.enqueue(db, "transcribe", video_id=video_id, max_attempts=1)
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status in ("failed", "dead")
        assert "faster-whisper" in (job.last_error or "")
        assert db.query(Transcript).filter(Transcript.video_id == video_id).count() == 0
        video = db.get(Video, video_id)
        assert video.status == "failed"


def test_gemini_job_uses_the_injected_client(monkeypatch):
    video_id = _make_video_with_audio()

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe_audio(self, audio_path, model=None):
            return [
                asr_service.ASRSegment(0.0, 2.0, "hello world", "en"),
                asr_service.ASRSegment(2.0, 4.0, "សូមស្វាគមន៍", "km"),
            ]

    monkeypatch.setattr(gemini_service, "GeminiClient", StubClient)
    with SessionLocal() as db:
        job = queue.enqueue(db, "gemini_transcribe", video_id=video_id)
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status == "succeeded"
        transcripts = db.query(Transcript).filter(Transcript.video_id == video_id).all()
        assert len(transcripts) == 1
        assert transcripts[0].source == "gemini"


def test_gemini_without_api_key_degrades_gracefully(monkeypatch):
    video_id = _make_video_with_audio()
    monkeypatch.setattr(gemini_service.settings, "gemini_api_key", None)
    with SessionLocal() as db:
        job = queue.enqueue(db, "gemini_transcribe", video_id=video_id, max_attempts=1)
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status in ("failed", "dead")
        assert "GEMINI_API_KEY" in (job.last_error or "")
        assert db.query(Transcript).filter(Transcript.video_id == video_id).count() == 0
