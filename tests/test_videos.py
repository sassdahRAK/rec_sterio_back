"""Videos: create, dedupe, listing, SSRF guard, detail, patch, delete, audio."""
from __future__ import annotations

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Job, Video
from app.services import storage
from tests.helpers import upload_video

URL = "https://www.youtube.com/watch?v=recsterio-test-001"


def test_create_video_returns_201_and_queues_download(client, auth_headers):
    resp = client.post(
        "/api/v1/videos",
        json={"source_url": URL, "title": "Test", "notes": "hello", "start_transcription": True},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    assert body["source_url"] == URL
    assert body["platform"] == "youtube"
    assert body["created_by"] == "super_admin:admin@test.local"

    jobs = client.get("/api/v1/jobs", headers=auth_headers).json()
    kinds = [j["kind"] for j in jobs["items"]]
    assert "download" in kinds
    assert jobs["items"][0]["status"] == "queued"


def test_video_appears_in_list(client, auth_headers):
    client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers)
    resp = client.get("/api/v1/videos", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["source_url"] == URL
    assert body["page"] == 1
    assert body["page_size"] == 20


def test_the_source_url_lock_makes_two_transactions_take_turns():
    """The mechanism behind the de-duplication, proven deterministically.

    The HTTP-level race below is too narrow to hit reliably in-process, so this
    pins the lock itself: a second transaction cannot obtain the same URL's
    lock until the first one commits.
    """
    import threading
    import time

    from app.routers.videos import _lock_on_source_url

    order: list[str] = []
    waited: list[float] = []

    def first() -> None:
        with SessionLocal() as db:
            _lock_on_source_url(db, URL)
            order.append("first-locked")
            time.sleep(0.4)  # stand in for the SELECT + INSERT the lock covers
            db.commit()

    def second() -> None:
        time.sleep(0.1)  # start while the first still holds it
        with SessionLocal() as db:
            started = time.monotonic()
            _lock_on_source_url(db, URL)
            waited.append(time.monotonic() - started)
            order.append("second-locked")
            db.commit()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # The second acquisition must have BLOCKED until the first committed.
    # Checking only the append order would pass without the lock too, since the
    # first thread appends before it sleeps.
    assert waited and waited[0] >= 0.2, f"second transaction did not wait: {waited}"
    assert order == ["first-locked", "second-locked"]


def test_simultaneous_creates_of_one_url_still_produce_one_video(client, auth_headers):
    """The double-tapped link button: two requests, one row.

    The de-duplication is a read followed by a write, so without the advisory
    lock both requests find nothing and both insert — and the worker then
    downloads and transcribes the same media twice. Threads here stand in for
    the two concurrent bot updates.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                client.post,
                "/api/v1/videos",
                json={"source_url": URL},
                headers=auth_headers,
            )
            for _ in range(2)
        ]
        results = [f.result() for f in futures]

    codes = sorted(r.status_code for r in results)
    assert codes == [201, 409], [r.text for r in results]
    with SessionLocal() as db:
        rows = db.execute(select(Video).where(Video.source_url == URL)).scalars().all()
        assert len(rows) == 1
        # And exactly one download job, not two.
        assert len(db.execute(select(Job)).scalars().all()) == 1


def test_duplicate_source_url_does_not_create_a_second_video(client, auth_headers):
    first = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers)
    assert first.status_code == 201
    second = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers)
    assert second.status_code == 409
    body = second.json()
    assert body["error"]["code"] == "conflict"
    # The already existing video is returned so clients can reuse it.
    assert body["video"]["id"] == first.json()["id"]

    with SessionLocal() as db:
        assert db.execute(select(Video)).scalars().all().__len__() == 1
    listing = client.get("/api/v1/videos", headers=auth_headers).json()
    assert listing["total"] == 1


def test_failed_video_can_be_recreated(client, auth_headers):
    created = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    with SessionLocal() as db:
        video = db.get(Video, created["id"])
        video.status = "failed"
        db.commit()
    again = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers)
    assert again.status_code == 201
    assert again.json()["id"] != created["id"]


def test_invalid_scheme_is_rejected(client, auth_headers):
    resp = client.post(
        "/api/v1/videos", json={"source_url": "file:///etc/passwd"}, headers=auth_headers
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_failed"
    assert body["error"]["details"][0]["field"] == "source_url"


def test_localhost_and_private_ips_are_rejected(client, auth_headers):
    for bad in (
        "http://localhost:8000/video.mp4",
        "http://127.0.0.1/video.mp4",
        "http://10.0.0.5/video.mp4",
        "http://192.168.1.10/video.mp4",
        "http://[::1]/video.mp4",
        "ftp://example.com/video.mp4",
    ):
        resp = client.post("/api/v1/videos", json={"source_url": bad}, headers=auth_headers)
        assert resp.status_code == 422, f"{bad} was accepted"
        assert resp.json()["error"]["code"] == "validation_failed"


def test_get_video_detail_includes_transcripts_and_jobs(client, auth_headers):
    video = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    resp = client.get(f"/api/v1/videos/{video['id']}", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == video["id"]
    assert body["transcripts"] == []
    assert len(body["jobs"]) == 1
    assert body["jobs"][0]["kind"] == "download"


def test_video_detail_does_not_expose_local_path(client, auth_headers):
    """The stored local_path is a server path and is not API surface at all now
    (even basename-only was a mapping for guessing the storage layout); media
    streams through /videos/{id}/media instead."""
    video = upload_video(client, auth_headers)
    resp = client.get(f"/api/v1/videos/{video['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert "local_path" not in resp.json()


def test_patch_video_updates_fields(client, auth_headers):
    """Metadata fields are editor-writable; `status` is not (the queue/runner
    and the ingestion routes own the lifecycle), so a PATCH carrying it is
    ignored for that field rather than poking the pipeline."""
    video = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    resp = client.patch(
        f"/api/v1/videos/{video['id']}",
        json={"title": "New title", "notes": "note", "status": "quarantined"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "New title"
    assert body["notes"] == "note"
    # The lifecycle state stays exactly what ingestion set it to.
    assert body["status"] == "pending"


def test_delete_video_removes_rows_and_media(client, auth_headers):
    video = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    resp = client.delete(f"/api/v1/videos/{video['id']}", headers=auth_headers)
    assert resp.status_code == 204
    with SessionLocal() as db:
        assert db.get(Video, video["id"]) is None
        assert db.execute(select(Job)).scalars().all() == []


def test_unknown_video_is_404_envelope(client, auth_headers):
    resp = client.get(
        "/api/v1/videos/00000000-0000-0000-0000-000000000000", headers=auth_headers
    )
    assert resp.status_code == 404
    body = resp.json()
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == "not_found"
    assert isinstance(body["error"]["message"], str)


def test_retry_reenqueues_download_job(client, auth_headers):
    video = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    # Creating the video already queued a download. Retrying a second time
    # would download the same file twice, so the API refuses — this is the
    # double-tapped Retry button.
    blocked = client.post(f"/api/v1/videos/{video['id']}/retry", headers=auth_headers)
    assert blocked.status_code == 409, blocked.text
    assert blocked.json()["error"]["code"] == "job_in_progress"
    assert blocked.json()["error"]["details"][0]["status"] == "queued"
    # Exactly one job exists: the refused attempt enqueued nothing.
    with SessionLocal() as db:
        assert len(db.execute(select(Job)).scalars().all()) == 1

    # Once the queued job has finished (here: failed), a retry is legitimate.
    with SessionLocal() as db:
        job = db.execute(select(Job)).scalars().one()
        job.status = "failed"
        db.commit()

    resp = client.post(f"/api/v1/videos/{video['id']}/retry", headers=auth_headers)
    assert resp.status_code == 202
    assert resp.json()["kind"] == "download"
    assert resp.json()["status"] == "queued"


def test_video_create_rate_limit(client, auth_headers):
    from app.config import settings
    from app.security import rate_limiter

    original = settings.video_create_rate_limit
    rate_limiter.reset()
    settings.video_create_rate_limit = 1
    try:
        first = client.post(
            "/api/v1/videos", json={"source_url": URL + "?a=1"}, headers=auth_headers
        )
        assert first.status_code == 201
        second = client.post(
            "/api/v1/videos", json={"source_url": URL + "?a=2"}, headers=auth_headers
        )
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "rate_limited"
        assert "Retry-After" in second.headers
    finally:
        settings.video_create_rate_limit = original


def test_file_size_accepts_a_multi_gigabyte_download(client, auth_headers):
    """A 2-hour YouTube video is ~7 GB, which overflows a 32-bit integer.

    The download handler writes `file_size` straight from the download result,
    so an `integer` column failed the UPDATE with `NumericValueOutOfRange` —
    poisoning the session and leaving the job orphaned at 'running'.
    """
    from sqlalchemy import text

    from app.db import SessionLocal, engine

    # The column must be wide enough before any insert is attempted.
    with engine.connect() as conn:
        column_type = conn.execute(
            text(
                "select data_type from information_schema.columns "
                "where table_name='videos' and column_name='file_size'"
            )
        ).scalar_one()
    assert column_type == "bigint", column_type

    seven_gb = 7_614_531_447
    assert seven_gb > 2_147_483_647  # would overflow the old integer column

    with SessionLocal() as db:
        video = Video(source_url=URL + "?big=1", platform="youtube", file_size=seven_gb)
        db.add(video)
        db.commit()
        db.refresh(video)
        assert video.file_size == seven_gb
        stored = db.execute(
            text("select file_size from videos where id=:i"), {"i": video.id}
        ).scalar_one()
        assert stored == seven_gb
        db.delete(video)
        db.commit()


# ------------------------------------------------------------------ audio


def test_audio_download_serves_the_wav_by_default(client, auth_headers):
    """Portal data cards link here without a format param; the wav is the default."""
    video = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    payload = b"RIFF" + b"\x00" * 128
    storage.audio_file(video["id"]).write_bytes(payload)

    resp = client.get(f"/api/v1/videos/{video['id']}/audio", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("audio/wav")
    assert (
        resp.headers["content-disposition"]
        == f'attachment; filename="{video["id"]}-audio.wav"'
    )
    assert resp.content == payload


def test_audio_download_serves_the_mp3_listening_copy(client, auth_headers):
    video = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    payload = b"ID3" + b"\x00" * 128
    storage.audio_mp3_file(video["id"]).write_bytes(payload)

    resp = client.get(
        f"/api/v1/videos/{video['id']}/audio", params={"format": "mp3"}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("audio/mpeg")
    assert (
        resp.headers["content-disposition"]
        == f'attachment; filename="{video["id"]}-audio.mp3"'
    )
    assert resp.content == payload


def test_audio_download_rejects_an_unknown_format(client, auth_headers):
    video = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    storage.audio_file(video["id"]).write_bytes(b"RIFF" + b"\x00" * 32)

    resp = client.get(
        f"/api/v1/videos/{video['id']}/audio", params={"format": "flac"}, headers=auth_headers
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "validation_failed"
    assert body["error"]["message"] == "format must be 'wav' or 'mp3'."


def test_audio_download_mp3_404_when_not_extracted(client, auth_headers):
    """The wav can exist without the mp3: extraction of the copy is best effort."""
    video = client.post("/api/v1/videos", json={"source_url": URL}, headers=auth_headers).json()
    storage.audio_file(video["id"]).write_bytes(b"RIFF" + b"\x00" * 32)

    resp = client.get(
        f"/api/v1/videos/{video['id']}/audio", params={"format": "mp3"}, headers=auth_headers
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["message"] == "MP3 has not been extracted for this video."
