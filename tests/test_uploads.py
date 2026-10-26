"""Direct file upload: ingestion must not depend on a platform serving a download.

TikTok and YouTube refuse most server-side requests, so an operator who already
has the file can POST it to /videos/upload and get the same pipeline behind it.
"""
from __future__ import annotations

import hashlib

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Job, Video


def _mp4_bytes() -> bytes:
    """A small, unique fake media payload. The upload path never decodes it."""
    return b"\x00\x00\x00\x18ftypmp42" + b"recsterio-upload-fixture" * 40


def test_upload_creates_a_downloaded_video_and_queues_audio(client, auth_headers):
    payload = _mp4_bytes()
    resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("clip.mp4", payload, "video/mp4")},
        data={"title": "Uploaded clip", "start_transcription": "true"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()

    assert body["status"] == "downloaded"
    assert body["progress"] == 100
    assert body["platform"] == "direct"
    assert body["file_size"] == len(payload)
    assert body["checksum_sha256"] == hashlib.sha256(payload).hexdigest()
    assert body["mime_type"] == "video/mp4"
    assert body["source_url"] == "upload://clip.mp4"

    jobs = client.get("/api/v1/jobs", headers=auth_headers).json()["items"]
    assert jobs[0]["kind"] == "extract_audio"
    assert jobs[0]["payload"]["chain"] == ["transcribe"]


def test_upload_without_transcription_queues_no_transcribe(client, auth_headers):
    resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("a.mp3", b"audio-bytes-x" * 10, "audio/mpeg")},
        data={"start_transcription": "false"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    jobs = client.get("/api/v1/jobs", headers=auth_headers).json()["items"]
    assert jobs[0]["payload"]["chain"] == []


def test_upload_same_bytes_is_rejected_as_a_duplicate(client, auth_headers):
    payload = _mp4_bytes()
    first = client.post(
        "/api/v1/videos/upload",
        files={"file": ("one.mp4", payload, "video/mp4")},
        headers=auth_headers,
    )
    assert first.status_code == 201

    second = client.post(
        "/api/v1/videos/upload",
        files={"file": ("two-different-name.mp4", payload, "video/mp4")},
        headers=auth_headers,
    )
    assert second.status_code == 409
    detail = second.json()
    assert detail["error"]["code"] == "conflict"
    assert detail["error"]["details"][0]["video_id"] == first.json()["id"]
    # The existing video is returned so the client does not have to re-upload.
    assert detail["video"]["id"] == first.json()["id"]

    with SessionLocal() as db:
        assert db.execute(select(Video)).scalars().all().__len__() == 1


def test_upload_rejects_an_unsupported_extension(client, auth_headers):
    resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("notes.exe", b"MZ-binary", "application/octet-stream")},
        headers=auth_headers,
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_media_type"
    with SessionLocal() as db:
        assert db.execute(select(Video)).scalars().all() == []


def test_upload_rejects_an_empty_file(client, auth_headers):
    resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("empty.mp4", b"", "video/mp4")},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_failed"
    with SessionLocal() as db:
        assert db.execute(select(Video)).scalars().all() == []
        assert db.execute(select(Job)).scalars().all() == []


def test_upload_falls_back_to_the_suffix_for_a_generic_content_type(client, auth_headers):
    resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("clip.mp4", b"payload-generic-type", "application/octet-stream")},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    # curl and many clients send octet-stream for everything; the dashboard
    # needs a real type for its <video> element.
    assert resp.json()["mime_type"] == "video/mp4"


def test_upload_requires_authentication(client):
    resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("clip.mp4", _mp4_bytes(), "video/mp4")},
    )
    assert resp.status_code == 401
