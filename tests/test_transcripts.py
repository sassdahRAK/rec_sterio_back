"""Transcripts: manual versioning, listing, generation enqueue."""
from __future__ import annotations

URL = "https://www.youtube.com/watch?v=recsterio-transcript-001"


def _create_video(client, headers) -> dict:
    resp = client.post("/api/v1/videos", json={"source_url": URL}, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_put_transcript_creates_version_1_then_2(client, auth_headers):
    video = _create_video(client, auth_headers)

    first = client.put(
        f"/api/v1/videos/{video['id']}/transcript",
        json={"content": "first version", "language": "km-en"},
        headers=auth_headers,
    )
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["version"] == 1
    assert body["source"] == "manual"
    assert body["created_by"] == "super_admin:admin@test.local"

    second = client.put(
        f"/api/v1/videos/{video['id']}/transcript",
        json={"content": "second version"},
        headers=auth_headers,
    )
    assert second.status_code == 201
    assert second.json()["version"] == 2
    assert second.json()["content"] == "second version"


def test_manual_edit_never_mutates_previous_versions(client, auth_headers):
    video = _create_video(client, auth_headers)
    client.put(
        f"/api/v1/videos/{video['id']}/transcript",
        json={"content": "original"},
        headers=auth_headers,
    )
    client.put(
        f"/api/v1/videos/{video['id']}/transcript",
        json={"content": "corrected"},
        headers=auth_headers,
    )

    versions = client.get(f"/api/v1/videos/{video['id']}/transcripts", headers=auth_headers)
    assert versions.status_code == 200
    items = versions.json()["items"]
    assert len(items) == 2
    # newest first
    assert items[0]["version"] == 2
    assert items[0]["content"] == "corrected"
    assert items[1]["version"] == 1
    assert items[1]["content"] == "original"


def test_current_transcript_is_the_latest_ready_version(client, auth_headers):
    video = _create_video(client, auth_headers)
    client.put(
        f"/api/v1/videos/{video['id']}/transcript",
        json={"content": "original"},
        headers=auth_headers,
    )
    client.put(
        f"/api/v1/videos/{video['id']}/transcript",
        json={"content": "corrected"},
        headers=auth_headers,
    )
    current = client.get(f"/api/v1/videos/{video['id']}/transcript", headers=auth_headers)
    assert current.status_code == 200
    assert current.json()["version"] == 2
    assert current.json()["content"] == "corrected"


def test_transcript_404_when_none_exists(client, auth_headers):
    video = _create_video(client, auth_headers)
    resp = client.get(f"/api/v1/videos/{video['id']}/transcript", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_put_transcript_with_empty_content_is_422(client, auth_headers):
    video = _create_video(client, auth_headers)
    resp = client.put(
        f"/api/v1/videos/{video['id']}/transcript",
        json={"content": "   "},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_failed"


def test_segments_endpoint(client, auth_headers):
    video = _create_video(client, auth_headers)
    client.put(
        f"/api/v1/videos/{video['id']}/transcript",
        json={
            "content": "hello world",
            "segments": [{"start": 0.0, "end": 1.5, "text": "hello", "language": "en"}],
        },
        headers=auth_headers,
    )
    resp = client.get(f"/api/v1/videos/{video['id']}/transcript/segments", headers=auth_headers)
    assert resp.status_code == 200
    segments = resp.json()["segments"]
    assert segments[0]["text"] == "hello"
    assert segments[0]["start"] == 0.0


def _drain_jobs(video_id: str) -> None:
    """Finish the queued jobs so the next generate call is not refused."""
    from sqlalchemy import select as _select

    from app.db import SessionLocal as _SessionLocal
    from app.models import Job as _Job

    with _SessionLocal() as db:
        for job in db.execute(_select(_Job).where(_Job.video_id == video_id)).scalars():
            if job.status in ("queued", "running"):
                job.status = "succeeded"
        db.commit()


def test_generate_enqueues_jobs_per_engine(client, auth_headers):
    video = _create_video(client, auth_headers)

    whisper = client.post(
        f"/api/v1/videos/{video['id']}/transcript/generate",
        json={"engine": "whisper"},
        headers=auth_headers,
    )
    assert whisper.status_code == 202
    jobs = whisper.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "extract_audio"
    assert jobs[0]["payload"]["chain"] == ["transcribe"]

    # A second tap while that run is still queued would start a rival ASR run
    # over the same media — the double-tapped "Transcribe again" button.
    double_tap = client.post(
        f"/api/v1/videos/{video['id']}/transcript/generate",
        json={"engine": "gemini"},
        headers=auth_headers,
    )
    assert double_tap.status_code == 409, double_tap.text
    assert double_tap.json()["error"]["code"] == "job_in_progress"

    _drain_jobs(video["id"])

    gemini = client.post(
        f"/api/v1/videos/{video['id']}/transcript/generate",
        json={"engine": "gemini"},
        headers=auth_headers,
    )
    assert gemini.status_code == 202
    assert gemini.json()["jobs"][0]["kind"] == "gemini_transcribe"

    _drain_jobs(video["id"])

    both = client.post(
        f"/api/v1/videos/{video['id']}/transcript/generate",
        json={"engine": "both"},
        headers=auth_headers,
    )
    assert both.status_code == 202
    kinds = {j["kind"] for j in both.json()["jobs"]}
    assert kinds == {"extract_audio", "gemini_transcribe"}


def test_generate_rejects_unknown_engine(client, auth_headers):
    video = _create_video(client, auth_headers)
    resp = client.post(
        f"/api/v1/videos/{video['id']}/transcript/generate",
        json={"engine": "nope"},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_failed"
