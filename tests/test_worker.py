"""Worker behaviour: job handlers, idempotency and graceful degradation."""
from __future__ import annotations

import wave
from pathlib import Path

from app.db import SessionLocal
from app.models import Job, Transcript, Video
from app.services import asr as asr_service
from app.services import downloader
from app.services import gemini as gemini_service
from app.services import storage
from app.services.downloader import DownloadResult
from app.worker import queue
from app.worker.runner import (
    handle_extract_audio,
    handle_gemini_transcribe,
    handle_transcribe,
    run_job,
)

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


def test_extract_only_leaves_the_video_downloaded(monkeypatch):
    """An upload with start_transcription=false must not strand the video.

    Regression: handle_extract_audio used to set status="transcribing"
    unconditionally. With an empty chain nothing follows, so the video sat in
    "transcribing" forever with no job left to advance it.
    """
    video_id = _make_video_with_audio()
    with SessionLocal() as db:
        job = queue.enqueue(db, "extract_audio", video_id=video_id, payload={"chain": []})
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status == "succeeded", job.last_error

    with SessionLocal() as db:
        video = db.get(Video, video_id)
        assert video.status == "downloaded"
        assert video.progress == 100
        # Nothing was chained, so no transcribe job may exist.
        assert (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "transcribe")
            .count()
            == 0
        )


def test_extract_with_a_transcribe_chain_moves_to_transcribing(monkeypatch):
    """The normal path still advances to "transcribing" and queues the next job."""
    video_id = _make_video_with_audio()
    with SessionLocal() as db:
        job = queue.enqueue(
            db, "extract_audio", video_id=video_id, payload={"chain": ["transcribe"]}
        )
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status == "succeeded", job.last_error

    with SessionLocal() as db:
        video = db.get(Video, video_id)
        assert video.status == "transcribing"
        nxt = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "transcribe")
            .one_or_none()
        )
        assert nxt is not None
        assert nxt.status == "queued"


def test_extract_job_also_calls_the_mp3_extractor(monkeypatch):
    """The extract job must produce the .mp3 listening copy, not just the wav."""
    video_id = _make_video_with_audio()
    captured = {}

    def fake_extract_mp3(media_path, out_path, timeout_seconds=None):
        captured["media"] = Path(media_path)
        captured["out"] = Path(out_path)
        out_path.write_bytes(b"ID3" + b"\x00" * 300)
        return out_path

    monkeypatch.setattr(asr_service, "extract_mp3", fake_extract_mp3)
    with SessionLocal() as db:
        job = queue.enqueue(db, "extract_audio", video_id=video_id, payload={"chain": []})
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status == "succeeded", job.last_error

    assert captured["media"].name == "media.mp4"
    assert captured["out"] == storage.audio_mp3_file(video_id)
    assert storage.audio_mp3_file(video_id).exists()


def test_extract_job_survives_an_mp3_extraction_failure(monkeypatch):
    """The wav is what ASR needs; a failed mp3 must not kill the job or chain."""
    video_id = _make_video_with_audio()

    def _boom(*args, **kwargs):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(asr_service, "extract_mp3", _boom)
    with SessionLocal() as db:
        job = queue.enqueue(
            db, "extract_audio", video_id=video_id, payload={"chain": ["transcribe"]}
        )
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status == "succeeded", job.last_error

    with SessionLocal() as db:
        video = db.get(Video, video_id)
        assert video.status == "transcribing"
        nxt = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "transcribe")
            .one_or_none()
        )
        assert nxt is not None
        assert nxt.status == "queued"
    assert not storage.audio_mp3_file(video_id).exists()


def test_handle_download_passes_the_per_user_cookiefile(monkeypatch, tmp_path):
    """A video ingested by a Telegram user downloads under their own cookies;
    any other actor ("system", the bot itself) uses the server defaults."""
    captured = {}

    def fake_download(source_url, video_id, on_progress=None, cookiefile=None):
        captured["cookiefile"] = cookiefile
        return DownloadResult(path=tmp_path / "media.mp4")

    monkeypatch.setattr(downloader, "download", fake_download)
    cookie_path = tmp_path / "424242.txt"
    cookie_path.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setattr(storage, "cookie_file", lambda uid: tmp_path / f"{uid}.txt")

    def _enqueue_download(created_by: str) -> str:
        with SessionLocal() as db:
            video = Video(
                source_url=URL, platform="youtube", status="pending", created_by=created_by
            )
            db.add(video)
            db.commit()
            db.refresh(video)
            job = queue.enqueue(db, "download", video_id=video.id, payload={"chain": []})
            db.commit()
            return str(job.id)

    job_id = _enqueue_download("telegram:424242")
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status == "succeeded", job.last_error
    assert captured["cookiefile"] == str(cookie_path)

    # An actor without a numeric Telegram suffix is not a user session.
    captured.clear()
    job_id = _enqueue_download("system")
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        assert job.status == "succeeded", job.last_error
    assert captured["cookiefile"] is None
