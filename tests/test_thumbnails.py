"""Local thumbnail cache: sniffing, access, serving and non-fatal failures.

Hermetic: every test replaces the HTTP fetch layer, so nothing reaches a CDN.
"""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.config import settings
from app.db import SessionLocal
from app.models import Folder, Job, Video
from app.services import downloader, storage
from app.services import thumbnail as thumbnail_service
from app.services.downloader import DownloadResult
from app.services.thumbnail import cache_from_url, public_thumbnail_url
from app.worker import queue, runner

JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-filler" * 8
PNG = b"\x89PNG\r\n\x1a\n" + b"png-filler" * 8
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"webp-filler" * 8

CDN_URL = "https://p16-common-sign.tiktokcdn.com/expired/frame.jpg"


# --------------------------------------------------------------- save/sniff


@pytest.mark.parametrize(
    "data, declared_type, expected_ext",
    [
        # The declared type lies; the bytes decide.
        (JPEG, "image/png", "jpg"),
        (PNG, "application/octet-stream", "png"),
        (WEBP, "image/jpeg", "webp"),
    ],
)
def test_save_thumbnail_uses_the_sniffed_extension(
    tmp_path, monkeypatch, data, declared_type, expected_ext
):
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    video_id = uuid4()

    path = storage.save_thumbnail(video_id, data, declared_type)

    assert path == tmp_path / str(video_id) / f"thumbnail.{expected_ext}"
    assert path.read_bytes() == data
    assert storage.thumbnail_file(video_id) == path


def test_save_thumbnail_rejects_oversized_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    video_id = uuid4()
    oversized = JPEG + b"0" * (storage.THUMBNAIL_MAX_BYTES)

    with pytest.raises(ValueError):
        storage.save_thumbnail(video_id, oversized, "image/jpeg")

    assert storage.thumbnail_file(video_id) is None


# --------------------------------------------------------- public URL helper


def _public_folder_and_video(client, auth_headers, visibility="public"):
    from tests.helpers import make_folder, upload_video

    folder = make_folder(client, auth_headers, visibility=visibility)
    video = upload_video(client, auth_headers, folder_id=folder["id"])
    return folder, video


def _set_thumbnail_url(video_id, url):
    with SessionLocal() as db:
        row = db.get(Video, UUID(str(video_id)))
        row.thumbnail_url = url
        db.commit()


def test_public_thumbnail_url_without_a_cached_frame(client, auth_headers):
    _, video = _public_folder_and_video(client, auth_headers)
    _set_thumbnail_url(video["id"], CDN_URL)

    with SessionLocal() as db:
        row = db.get(Video, UUID(video["id"]))
        assert public_thumbnail_url(row) == CDN_URL


def test_public_thumbnail_url_for_a_cached_public_video(client, auth_headers):
    folder, video = _public_folder_and_video(client, auth_headers)
    _set_thumbnail_url(video["id"], CDN_URL)
    storage.save_thumbnail(video["id"], JPEG, "image/jpeg")

    with SessionLocal() as db:
        row = db.get(Video, UUID(video["id"]))
        folder_row = db.get(Folder, UUID(folder["id"]))
        expected = f"/api/v1/public/videos/{video['id']}/thumbnail"
        # Both call styles (explicit folder, and discovered through the session).
        assert public_thumbnail_url(row, folder_row) == expected
        assert public_thumbnail_url(row) == expected


def test_public_thumbnail_url_keeps_the_cdn_for_a_private_folder(client, auth_headers):
    folder, video = _public_folder_and_video(client, auth_headers, visibility="private")
    _set_thumbnail_url(video["id"], CDN_URL)
    storage.save_thumbnail(video["id"], JPEG, "image/jpeg")

    with SessionLocal() as db:
        row = db.get(Video, UUID(video["id"]))
        assert public_thumbnail_url(row) == CDN_URL


# ------------------------------------------------------------------ endpoint


def test_thumbnail_endpoint_serves_a_cached_public_frame(client, auth_headers):
    _, video = _public_folder_and_video(client, auth_headers)
    storage.save_thumbnail(video["id"], PNG, "image/png")

    # Anonymous caller: no Authorization header.
    resp = client.get(f"/api/v1/public/videos/{video['id']}/thumbnail")

    assert resp.status_code == 200, resp.text
    assert resp.content == PNG
    assert resp.headers["content-type"].startswith("image/png")
    assert resp.headers["cache-control"] == "public, max-age=604800, immutable"
    assert resp.headers.get("etag")


def test_thumbnail_endpoint_hides_a_private_video_from_anonymous(client, auth_headers):
    _, video = _public_folder_and_video(client, auth_headers, visibility="private")
    storage.save_thumbnail(video["id"], PNG, "image/png")

    resp = client.get(f"/api/v1/public/videos/{video['id']}/thumbnail")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["message"] == "Video not found."
    text = resp.text
    assert str(settings.media_root_path) not in text
    assert "thumbnail" not in text.lower()
    assert "storage" not in text.lower()


def test_thumbnail_endpoint_reports_a_missing_cache_without_leaking(client, auth_headers):
    _, video = _public_folder_and_video(client, auth_headers)

    resp = client.get(f"/api/v1/public/videos/{video['id']}/thumbnail")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "thumbnail_not_ready"
    assert str(settings.media_root_path) not in resp.text


# --------------------------------------------------------------- fetch layer


def test_cache_from_url_writes_a_200_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    monkeypatch.setattr(thumbnail_service, "assert_fetch_url_safe", lambda url: url)
    monkeypatch.setattr(
        thumbnail_service,
        "http_bytes",
        lambda *a, **k: (200, {"content-type": "image/jpeg"}, JPEG),
    )
    video_id = uuid4()

    path = cache_from_url(video_id, CDN_URL)

    assert path is not None
    assert path.read_bytes() == JPEG
    assert storage.thumbnail_file(video_id) == path


def test_cache_from_url_returns_none_on_a_non_200(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    monkeypatch.setattr(thumbnail_service, "assert_fetch_url_safe", lambda url: url)
    monkeypatch.setattr(
        thumbnail_service,
        "http_bytes",
        lambda *a, **k: (403, {"content-type": "text/html"}, b"<html>denied</html>"),
    )
    video_id = uuid4()

    assert cache_from_url(video_id, CDN_URL) is None
    assert storage.thumbnail_file(video_id) is None


def test_cache_from_url_returns_none_for_a_rejected_url(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    # The real guard runs (and must refuse an IP literal without any DNS call).
    video_id = uuid4()

    assert cache_from_url(video_id, "http://127.0.0.1/frame.jpg") is None
    assert storage.thumbnail_file(video_id) is None


def test_cache_from_url_returns_none_on_a_transport_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    monkeypatch.setattr(thumbnail_service, "assert_fetch_url_safe", lambda url: url)

    def _boom(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(thumbnail_service, "http_bytes", _boom)

    assert cache_from_url(uuid4(), CDN_URL) is None


# -------------------------------------------------------------------- worker

WORKER_URL = "https://www.youtube.com/watch?v=recsterio-thumbnail-001"


def test_download_job_survives_a_thumbnail_cache_failure(monkeypatch, tmp_path):
    def fake_download(source_url, video_id, on_progress=None, cookiefile=None):
        return DownloadResult(path=tmp_path / "media.mp4", thumbnail_url=CDN_URL)

    monkeypatch.setattr(downloader, "download", fake_download)

    def _boom(*args, **kwargs):
        raise RuntimeError("thumbnail exploded")

    monkeypatch.setattr(runner.thumbnail, "cache_from_url", _boom)

    with SessionLocal() as db:
        video = Video(source_url=WORKER_URL, platform="youtube", status="pending")
        db.add(video)
        db.commit()
        db.refresh(video)
        job = queue.enqueue(db, "download", video_id=video.id, payload={"chain": []})
        db.commit()
        job_id = str(job.id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        runner.run_job(db, job)
        assert job.status == "succeeded", job.last_error
        assert db.get(Video, job.video_id).status == "downloaded"
