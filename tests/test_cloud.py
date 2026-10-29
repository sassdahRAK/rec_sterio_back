"""Personal-cloud storage (`/cloud`): connect, cloud upload, streaming.

The provider's HTTP layer is faked by monkeypatching the small functions in
`app.services.cloud_providers`, so the suite never reaches Google or Microsoft
and never needs real OAuth apps. The routes, the schema, the model wiring and
the streaming branch in `videos.py` are all exercised for real.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select

from app.db import SessionLocal
from app.models import DriveLink, Job, Video
from app.services import cloud_providers
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD
from tests.helpers import ensure_user, upload_video, user_headers

API = "/api/v1"

GOOGLE = "google"
ONEDRIVE = "onedrive"


def _admin(client) -> dict:
    resp = client.post(
        f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


@pytest.fixture
def drive_config(monkeypatch):
    """Pretend both OAuth apps are configured, without touching the real env.

    Drive tokens must be stored under an encryption key (an unset key refuses
    the connect flow in production), so the tests configure a real one and
    exercise the actual AESGCM encrypt/decrypt path with it.
    """
    import base64

    from app.config import settings

    monkeypatch.setattr(settings, "google_client_id", "g-id")
    monkeypatch.setattr(settings, "google_client_secret", "g-secret")
    monkeypatch.setattr(settings, "onedrive_client_id", "m-id")
    monkeypatch.setattr(settings, "onedrive_client_secret", "m-secret")
    monkeypatch.setattr(settings, "drive_oauth_redirect", "http://localhost/cb")
    monkeypatch.setattr(
        settings, "cloud_token_key", base64.urlsafe_b64encode(b"x" * 32).decode().rstrip("=")
    )
    return settings


def _fake_connect(monkeypatch, provider: str, email: str):
    monkeypatch.setattr(
        cloud_providers,
        "exchange_code",
        lambda p, code, redirect: {
            "refresh_token": f"refresh-{p}-{uuid4().hex[:8]}",
            "account_email": email,
            "account_subject": None,
        },
    )


def _fake_refresh_and_upload(monkeypatch, file_id: str = "file-123"):
    """refresh_access_token + create_upload_target return canned values."""
    monkeypatch.setattr(cloud_providers, "refresh_access_token", lambda p, rt: f"access-{p}")
    monkeypatch.setattr(
        cloud_providers,
        "create_upload_target",
        lambda p, access, name, mime, size: cloud_providers.UploadTarget(
            upload_url=f"https://uploads.example/{p}/{file_id}",
            upload_headers={"Content-Type": mime},
            expires_at=None,
        ),
    )
    # Prevent _verify_drive_file from making real HTTP calls during tests.
    monkeypatch.setattr("app.routers.cloud._verify_drive_file", lambda p, a, f: None)


def _fake_download(monkeypatch, body: bytes = b"\x00\x00\x00\x18ftypmp42cloud"):
    """open_download yields a canned CloudRead of `body`."""
    def _open(provider, access, file_id, range_header=None):
        return cloud_providers.CloudRead(
            status=206 if range_header else 200,
            headers={"Content-Type": "video/mp4"},
            size=len(body),
            body=iter([body]),
        )

    monkeypatch.setattr(cloud_providers, "open_download", _open)


def _connect(client, admin, provider, email, monkeypatch):
    _fake_connect(monkeypatch, provider, email)
    resp = client.post(
        f"{API}/cloud/links", json={"provider": provider, "code": "code-abc"}, headers=admin
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ------------------------------------------------------------------ connect


def test_providers_reflect_whether_the_oauth_apps_are_configured(client, drive_config):
    body = client.get(f"{API}/cloud/providers").json()
    assert body["google"] is True
    assert body["onedrive"] is True


def _unconfigure(monkeypatch, settings, provider: str) -> None:
    """Hide one OAuth app (id + secret) exactly the way the deployment would:
    `providers_available` must then report it False and publish no client id."""
    if provider == ONEDRIVE:
        monkeypatch.setattr(settings, "onedrive_client_id", None)
        monkeypatch.setattr(settings, "onedrive_client_secret", None)
        return
    monkeypatch.setattr(settings, "google_client_id", None)
    monkeypatch.setattr(settings, "google_client_secret", None)
    monkeypatch.setattr(settings, "drive_google_client_id", None)
    monkeypatch.setattr(settings, "drive_google_client_secret", None)


def test_google_client_id_is_published_when_onedrive_is_not_configured(
    client, drive_config, monkeypatch
):
    _unconfigure(monkeypatch, drive_config, ONEDRIVE)
    body = client.get(f"{API}/cloud/providers").json()
    assert body["google"] is True
    assert body["google_client_id"] == "g-id"
    assert body["onedrive"] is False
    assert body["onedrive_client_id"] is None


def test_onedrive_client_id_is_published_when_google_is_not_configured(
    client, drive_config, monkeypatch
):
    _unconfigure(monkeypatch, drive_config, GOOGLE)
    body = client.get(f"{API}/cloud/providers").json()
    assert body["onedrive"] is True
    assert body["onedrive_client_id"] == "m-id"
    assert body["google"] is False
    assert body["google_client_id"] is None


def test_onedrive_scope_covers_identity_read():
    """`/me` needs User.Read on top of the app folder scope, or connect 403s."""
    assert "https://graph.microsoft.com/Files.ReadWrite.AppFolder" in cloud_providers.ONEDRIVE_SCOPE
    assert "https://graph.microsoft.com/User.Read" in cloud_providers.ONEDRIVE_SCOPE
    assert cloud_providers.ONEDRIVE_SCOPE.startswith("offline_access ")


def test_a_signed_in_user_connects_a_drive_and_lists_it(client, drive_config, monkeypatch):
    admin = _admin(client)
    link = _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    assert link["provider"] == GOOGLE
    assert link["account_email"] == "me@gmail.com"
    assert "refresh_token" not in link

    listed = client.get(f"{API}/cloud/links", headers=admin).json()
    assert [l["provider"] for l in listed] == [GOOGLE]


def test_reconnecting_replaces_the_tokens_not_duplicate_rows(client, drive_config, monkeypatch):
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "old@gmail.com", monkeypatch)
    link2 = _connect(client, admin, GOOGLE, "new@gmail.com", monkeypatch)
    with SessionLocal() as db:
        rows = db.execute(select(DriveLink)).scalars().all()
    assert len(rows) == 1
    assert rows[0].account_email == "new@gmail.com"
    assert link2["account_email"] == "new@gmail.com"


def test_disconnect_removes_the_link_and_marks_videos_unavailable(client, drive_config, monkeypatch):
    admin = _admin(client)
    link = _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    # A completed cloud upload tied to this link (provider HTTP faked up front).
    _fake_refresh_and_upload(monkeypatch)
    _fake_download(monkeypatch)
    start = client.post(
        f"{API}/cloud/uploads",
        json={"backend": GOOGLE, "filename": "clip.mp4"},
        headers=admin,
    )
    assert start.status_code == 201, start.text
    vid = start.json()["video_id"]
    done = client.post(
        f"{API}/cloud/uploads/{vid}/complete",
        json={"provider_file_id": "file-9"},
        headers=admin,
    )
    assert done.status_code == 200, done.text

    resp = client.request("DELETE", f"{API}/cloud/links/{GOOGLE}", headers=admin)
    assert resp.status_code == 204, resp.text

    # The card now reports the unit unavailable, not a crash.
    cards = client.get(f"{API}/dataset/cards", headers=admin).json()
    mine = next(c for c in cards["items"] if c["video_id"] == vid)
    assert mine["storage_backend"] == GOOGLE
    assert mine["storage_available"] is False


def test_a_guest_cannot_connect_or_list(client, drive_config):
    assert client.get(f"{API}/cloud/links").status_code == 401
    assert (
        client.post(
            f"{API}/cloud/links", json={"provider": GOOGLE, "code": "x"}
        ).status_code
        == 401
    )


# --------------------------------------------------------------- cloud upload


def test_start_cloud_upload_returns_a_session_for_the_connected_drive(
    client, drive_config, monkeypatch
):
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    _fake_refresh_and_upload(monkeypatch, file_id="file-up")

    resp = client.post(
        f"{API}/cloud/uploads",
        json={"backend": GOOGLE, "filename": "clip.mp4"},
        headers=admin,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["upload_url"].startswith("https://uploads.example/google/")
    assert body["video_id"]

    with SessionLocal() as db:
        video = db.get(Video, body["video_id"])
    assert video.storage_backend == GOOGLE
    assert video.status == "uploading"
    assert video.drive_file_id is None


def test_cloud_upload_needs_a_connected_drive(client, drive_config):
    admin = _admin(client)
    resp = client.post(
        f"{API}/cloud/uploads", json={"backend": ONEDRIVE, "filename": "a.mp4"}, headers=admin
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "drive_not_connected"


def test_complete_cloud_upload_runs_the_pipeline(client, drive_config, monkeypatch):
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    _fake_refresh_and_upload(monkeypatch)
    _fake_download(monkeypatch, body=b"\x00\x00\x00\x18ftypmp42" + b"x" * 64)

    start = client.post(
        f"{API}/cloud/uploads", json={"backend": GOOGLE, "filename": "clip.mp4"}, headers=admin
    ).json()

    done = client.post(
        f"{API}/cloud/uploads/{start['video_id']}/complete",
        json={"provider_file_id": "file-1", "size": 80},
        headers=admin,
    )
    assert done.status_code == 200, done.text
    body = done.json()
    assert body["status"] == "downloaded"
    assert body["storage_backend"] == GOOGLE

    # The extract_audio job is queued with the transcribe chain behind it.
    with SessionLocal() as db:
        jobs = db.execute(
            select(Job).where(Job.video_id == start["video_id"])
        ).scalars().all()
    assert any(j.kind == "extract_audio" for j in jobs)


def test_complete_twice_is_a_conflict(client, drive_config, monkeypatch):
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    _fake_refresh_and_upload(monkeypatch)
    _fake_download(monkeypatch)
    start = client.post(
        f"{API}/cloud/uploads", json={"backend": GOOGLE}, headers=admin
    ).json()
    client.post(
        f"{API}/cloud/uploads/{start['video_id']}/complete",
        json={"provider_file_id": "f"},
        headers=admin,
    )
    again = client.post(
        f"{API}/cloud/uploads/{start['video_id']}/complete",
        json={"provider_file_id": "f"},
        headers=admin,
    )
    assert again.status_code == 409, again.text


def test_only_the_owner_can_complete_their_upload(client, drive_config, monkeypatch):
    admin = _admin(client)
    owner_email = ensure_user("owner")
    owner = user_headers(owner_email)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    _fake_refresh_and_upload(monkeypatch)
    _fake_download(monkeypatch)

    start = client.post(
        f"{API}/cloud/uploads", json={"backend": GOOGLE}, headers=owner
    )
    assert start.status_code == 409, start.text  # the owner never connected a drive

    # Connect as the owner and start a real one.
    owner_admin = _admin(client)  # not the owner; use owner session
    _connect(client, owner, GOOGLE, "owner@gmail.com", monkeypatch)
    start = client.post(f"{API}/cloud/uploads", json={"backend": GOOGLE}, headers=owner).json()
    stranger = user_headers(ensure_user("stranger"))
    denied = client.post(
        f"{API}/cloud/uploads/{start['video_id']}/complete",
        json={"provider_file_id": "f"},
        headers=stranger,
    )
    assert denied.status_code == 403, denied.text


# ------------------------------------------------------------------ streaming


def test_cloud_video_streams_through_the_media_route(client, drive_config, monkeypatch):
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    _fake_refresh_and_upload(monkeypatch)
    blob = b"\x00\x00\x00\x18ftypmp42" + b"z" * 128
    _fake_download(monkeypatch, body=blob)

    start = client.post(
        f"{API}/cloud/uploads", json={"backend": GOOGLE, "filename": "clip.mp4"}, headers=admin
    ).json()
    client.post(
        f"{API}/cloud/uploads/{start['video_id']}/complete",
        json={"provider_file_id": "file-1"},
        headers=admin,
    )

    resp = client.get(f"{API}/videos/{start['video_id']}/media", headers=admin)
    assert resp.status_code == 200, resp.text
    assert resp.content == blob


def test_a_dead_drive_reports_unavailable_not_a_500(client, drive_config, monkeypatch):
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    _fake_refresh_and_upload(monkeypatch)
    _fake_download(monkeypatch)

    start = client.post(
        f"{API}/cloud/uploads", json={"backend": GOOGLE}, headers=admin
    ).json()
    client.post(
        f"{API}/cloud/uploads/{start['video_id']}/complete",
        json={"provider_file_id": "file-1"},
        headers=admin,
    )

    # Break the grant: refresh now fails.
    def _boom(provider, rt):
        from app.errors import AppError

        raise AppError(400, "cloud_auth_rejected", "reconnect")

    monkeypatch.setattr(cloud_providers, "refresh_access_token", _boom)
    resp = client.get(f"{API}/videos/{start['video_id']}/media", headers=admin)
    assert resp.status_code in (400, 404), resp.status_code
