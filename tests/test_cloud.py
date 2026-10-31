"""Personal-cloud storage (`/cloud`): connect, cloud upload, streaming.

The provider's HTTP layer is faked by monkeypatching the small functions in
`app.services.cloud_providers`, so the suite never reaches Google or Microsoft
and never needs real OAuth apps. The routes, the schema, the model wiring and
the streaming branch in `videos.py` are all exercised for real.
"""
from __future__ import annotations

import json
import logging
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.db import SessionLocal
from app.models import AdminUser, DriveLink, Job, Video
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


_GOOGLE_GRANT_SCOPE = f"openid email {cloud_providers.DRIVE_FILE_SCOPE}"
_ONEDRIVE_GRANT_SCOPE = "Files.ReadWrite.AppFolder User.Read offline_access"
_DEFAULT_SCOPE = object()


def _fake_connect(monkeypatch, provider: str, email: str, scope=_DEFAULT_SCOPE):
    """Stub `exchange_code` with a `CloudGrant`, like the real exchange returns.

    `scope` defaults to the properly-granted scope for the provider; pass
    `None` to simulate a token response with no `scope` field, or a narrowed
    string to simulate the user unchecking Drive at the consent screen.
    """
    if scope is _DEFAULT_SCOPE:
        scope = _GOOGLE_GRANT_SCOPE if provider == GOOGLE else _ONEDRIVE_GRANT_SCOPE
    monkeypatch.setattr(
        cloud_providers,
        "exchange_code",
        lambda p, code, redirect: cloud_providers.CloudGrant(
            refresh_token=f"refresh-{p}-{uuid4().hex[:8]}",
            account_email=email,
            account_subject=None,
            scope=scope,
        ),
    )


def _fake_refresh_and_upload(monkeypatch, file_id: str = "file-123"):
    """refresh_access_token + create_upload_target return canned values."""
    monkeypatch.setattr(cloud_providers, "refresh_access_token", lambda p, rt: f"access-{p}")
    monkeypatch.setattr(
        cloud_providers,
        "create_upload_target",
        lambda p, access, name, mime, size, parent_id=None: cloud_providers.UploadTarget(
            upload_url=f"https://uploads.example/{p}/{file_id}",
            upload_headers={"Content-Type": mime},
            expires_at=None,
        ),
    )
    # Folder resolution must never reach Google from the suite.
    monkeypatch.setattr(cloud_providers, "ensure_app_folder", lambda *a, **k: None)
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


def test_connect_rejects_a_grant_that_lacks_the_drive_scope(client, drive_config, monkeypatch):
    """The live bug: consent narrowed to openid+email connected fine, then every
    Drive call 403'd. Now the connect is refused and nothing is stored.

    Fails on the old code (which ignored the token response's `scope`): it
    returned 201 and created a DriveLink row.
    """
    # A fresh account: `clean_database` wipes non-admin users, and the fixture
    # keeps drive_links, so this user is guaranteed to have no link.
    email = ensure_user("scopecheck")
    headers = user_headers(email)
    narrowed = "openid https://www.googleapis.com/auth/userinfo.email"
    _fake_connect(monkeypatch, GOOGLE, "me@gmail.com", scope=narrowed)

    resp = client.post(
        f"{API}/cloud/links", json={"provider": GOOGLE, "code": "code-abc"}, headers=headers
    )

    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["code"] == "drive_scope_missing"
    assert cloud_providers.DRIVE_FILE_SCOPE in error["message"]
    details = error["details"][0]
    assert details["provider"] == GOOGLE
    assert details["granted_scopes"] == narrowed.split()
    assert details["missing_scope"] == cloud_providers.DRIVE_FILE_SCOPE

    # Nothing persisted, and the links list stays empty for this account.
    with SessionLocal() as db:
        user = db.execute(
            select(AdminUser).where(AdminUser.email == email)
        ).scalar_one()
        assert db.execute(
            select(DriveLink).where(DriveLink.user_id == user.id)
        ).scalars().all() == []
    assert client.get(f"{API}/cloud/links", headers=headers).json() == []


def test_connect_accepts_a_grant_that_includes_the_drive_scope(client, drive_config, monkeypatch):
    """A properly-scoped consent still connects exactly as before."""
    admin = _admin(client)
    link = _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    assert link["provider"] == GOOGLE
    with SessionLocal() as db:
        rows = db.execute(select(DriveLink)).scalars().all()
    assert len(rows) == 1
    assert rows[0].account_email == "me@gmail.com"


def test_connect_allows_a_token_response_with_no_scope_field(
    client, drive_config, monkeypatch, caplog
):
    """Some providers/tenants omit `scope`; allow it but leave a warning.

    Refusing would break an otherwise usable grant, so the connect proceeds
    and the omission is logged at WARNING.
    """
    admin = _admin(client)
    _fake_connect(monkeypatch, GOOGLE, "me@gmail.com", scope=None)

    with caplog.at_level(logging.WARNING, logger="recsterio.cloud"):
        resp = client.post(
            f"{API}/cloud/links", json={"provider": GOOGLE, "code": "code-abc"}, headers=admin
        )

    assert resp.status_code == 201, resp.text
    assert resp.json()["account_email"] == "me@gmail.com"
    assert any("carried no scope" in r.message for r in caplog.records)
    with SessionLocal() as db:
        assert len(db.execute(select(DriveLink)).scalars().all()) == 1


def test_onedrive_grant_without_the_app_folder_scope_is_rejected(
    client, drive_config, monkeypatch
):
    """The analogous OneDrive check: User.Read alone is not enough."""
    email = ensure_user("scopecheck-od")
    headers = user_headers(email)
    _fake_connect(
        monkeypatch, ONEDRIVE, "me@outlook.com", scope="offline_access User.Read"
    )

    resp = client.post(
        f"{API}/cloud/links", json={"provider": ONEDRIVE, "code": "code-abc"}, headers=headers
    )

    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["code"] == "drive_scope_missing"
    assert "Files.ReadWrite.AppFolder" in error["message"]
    assert error["details"][0]["missing_scope"] == "Files.ReadWrite.AppFolder"
    with SessionLocal() as db:
        user = db.execute(
            select(AdminUser).where(AdminUser.email == email)
        ).scalar_one()
        assert db.execute(
            select(DriveLink).where(DriveLink.user_id == user.id)
        ).scalars().all() == []


def test_onedrive_grant_with_the_full_url_scope_is_accepted(
    client, drive_config, monkeypatch
):
    """Microsoft may echo the full Graph URL; the check accepts either form."""
    email = ensure_user("scopecheck-od-url")
    headers = user_headers(email)
    _fake_connect(
        monkeypatch,
        ONEDRIVE,
        "me@outlook.com",
        scope="https://graph.microsoft.com/Files.ReadWrite.AppFolder https://graph.microsoft.com/User.Read",
    )

    resp = client.post(
        f"{API}/cloud/links", json={"provider": ONEDRIVE, "code": "code-abc"}, headers=headers
    )

    assert resp.status_code == 201, resp.text
    assert resp.json()["provider"] == ONEDRIVE
    with SessionLocal() as db:
        user = db.execute(
            select(AdminUser).where(AdminUser.email == email)
        ).scalar_one()
        assert len(
            db.execute(
                select(DriveLink).where(DriveLink.user_id == user.id)
            ).scalars().all()
        ) == 1


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


def _fake_google_folder_and_upload(
    monkeypatch,
    *,
    existing_folder_id: str | None = None,
    created_folder_id: str = "folder-created",
    lookup_status: int = 200,
    lookup_raises: bool = False,
    creation_status: int = 200,
):
    """Serve the real ensure_app_folder + create_upload_target over fake HTTP.

    Records every provider call so a test can inspect the upload-session
    metadata (`parents`) exactly as Google would receive it.
    """
    from app.errors import AppError

    calls = {"json": [], "bytes": []}

    def _json(method, url, *, headers=None, json_body=None, data=None, timeout=30.0):
        calls["json"].append({"method": method, "url": url, "json": json_body})
        if method == "GET":
            if lookup_raises:
                raise AppError(502, "upstream_unreachable", "Could not reach googleapis.")
            if lookup_status >= 400:
                return lookup_status, {}, {"error": {"code": lookup_status}}
            files = (
                [{"id": existing_folder_id, "name": "RecSterio"}]
                if existing_folder_id
                else []
            )
            return 200, {}, {"files": files}
        # Folder creation.
        if creation_status >= 400:
            return creation_status, {}, {"error": {"code": creation_status}}
        return 200, {}, {"id": created_folder_id}

    def _bytes(method, url, *, headers=None, body=None, stream=False):
        calls["bytes"].append(
            {"method": method, "url": url, "headers": headers, "body": body}
        )
        assert stream is False
        return 200, {"Location": "https://uploads.example/session-1"}, b""

    monkeypatch.setattr(cloud_providers, "http_json", _json)
    monkeypatch.setattr(cloud_providers, "http_bytes", _bytes)
    return calls


def _start_google_upload(client, admin, monkeypatch) -> tuple:
    """Stub the token refresh and start a Google upload on a connected drive."""
    monkeypatch.setattr(cloud_providers, "refresh_access_token", lambda p, rt: "access-google")
    resp = client.post(
        f"{API}/cloud/uploads",
        json={"backend": GOOGLE, "filename": "clip.mp4"},
        headers=admin,
    )
    return resp


def test_cloud_upload_uses_the_existing_app_folder_as_parent(
    client, drive_config, monkeypatch
):
    """D2: the file goes INSIDE the app folder, with a real file name.

    Regression (live run): the metadata was
    `name="RecSterio/87e06fdbf9f2-f64b8a89.mp4"`, `parents=[<My Drive root>]`
    — a flat file in root, and no `RecSterio` folder existed. On the old code
    there is no `parents` key and the name contains a slash, so this fails.
    """
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    monkeypatch.setattr(cloud_providers, "refresh_access_token", lambda p, rt: "access-google")
    calls = _fake_google_folder_and_upload(monkeypatch, existing_folder_id="folder-old")

    resp = _start_google_upload(client, admin, monkeypatch)
    assert resp.status_code == 201, resp.text
    assert resp.json()["upload_url"] == "https://uploads.example/session-1"

    # One lookup, no creation: the folder is reused, never duplicated.
    assert [c["method"] for c in calls["json"]] == ["GET"]
    lookup_q = parse_qs(urlparse(calls["json"][0]["url"]).query)["q"][0]
    assert "mimeType='application/vnd.google-apps.folder'" in lookup_q
    assert "name='RecSterio'" in lookup_q
    assert "trashed=false" in lookup_q

    session = calls["bytes"][0]
    metadata = json.loads(session["body"])
    assert metadata["parents"] == ["folder-old"]
    assert "/" not in metadata["name"]
    assert metadata["name"].endswith(".mp4")
    assert metadata["mimeType"] == "video/mp4"


def test_cloud_upload_creates_the_app_folder_once_when_missing(
    client, drive_config, monkeypatch
):
    """An empty lookup creates one folder, then the upload is parented to it."""
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    monkeypatch.setattr(cloud_providers, "refresh_access_token", lambda p, rt: "access-google")
    calls = _fake_google_folder_and_upload(monkeypatch)

    resp = _start_google_upload(client, admin, monkeypatch)
    assert resp.status_code == 201, resp.text

    assert [c["method"] for c in calls["json"]] == ["GET", "POST"]
    creation = calls["json"][1]
    assert creation["json"] == {
        "name": "RecSterio",
        "mimeType": "application/vnd.google-apps.folder",
    }

    metadata = json.loads(calls["bytes"][0]["body"])
    assert metadata["parents"] == ["folder-created"]
    assert "/" not in metadata["name"]


@pytest.mark.parametrize("failure", ["status", "transport"])
def test_cloud_upload_falls_back_to_root_when_folder_lookup_fails(
    client, drive_config, monkeypatch, caplog, failure
):
    """Folder trouble must never cost the upload: root + a warning instead."""
    admin = _admin(client)
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    monkeypatch.setattr(cloud_providers, "refresh_access_token", lambda p, rt: "access-google")
    calls = _fake_google_folder_and_upload(
        monkeypatch,
        lookup_status=500 if failure == "status" else 200,
        lookup_raises=failure == "transport",
    )

    with caplog.at_level(logging.WARNING, logger="recsterio.cloud"):
        resp = _start_google_upload(client, admin, monkeypatch)

    assert resp.status_code == 201, resp.text
    # Only the failed lookup: no creation attempt on top of a bad lookup.
    assert [c["method"] for c in calls["json"]] == ["GET"]
    metadata = json.loads(calls["bytes"][0]["body"])
    assert "parents" not in metadata
    assert "/" not in metadata["name"]
    # The session itself still opened, and the reason is in the log.
    assert calls["bytes"][0]["method"] == "POST"
    assert "My Drive root" in caplog.text


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


class _DriveBody:
    """Minimal streaming body: what `http_bytes(..., stream=True)` returns."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self):
        self.closed = True


def _fake_provider_http(
    monkeypatch,
    blob: bytes,
    *,
    probe_mime: str = "video/mp4",
    download_mime: str = "video/quicktime",
    probe_has_size: bool = True,
):
    """Serve `blob` the way a drive does, including real Range semantics.

    The probe (`Range: bytes=0-0`) answers first, then each download answers
    either a 206 with the sliced bytes or a 200 with the whole file. Header
    keys are lowercase, exactly like `dict(httpx.Response.headers)`, so this
    also exercises the case-insensitive lookups. `probe_has_size=False`
    simulates a probe that cannot report a size.
    """

    def _bytes(method, url, *, headers=None, body=None, stream=False):
        assert stream is True
        sent = headers or {}
        rng = sent.get("Range")
        if rng == "bytes=0-0":
            probe_headers = {"content-type": probe_mime}
            if probe_has_size:
                probe_headers.update(
                    {
                        "content-range": f"bytes 0-0/{len(blob)}",
                        "content-length": "1",
                        "accept-ranges": "bytes",
                    }
                )
            return 206, probe_headers, _DriveBody([blob[:1]])
        if rng:
            start_raw, end_raw = rng.split("=", 1)[1].split("-", 1)
            start = int(start_raw)
            end = min(int(end_raw), len(blob) - 1)
            chunk = blob[start : end + 1]
            return (
                206,
                {
                    "content-type": download_mime,
                    "content-range": f"bytes {start}-{end}/{len(blob)}",
                    "content-length": str(len(chunk)),
                    "accept-ranges": "bytes",
                },
                _DriveBody([chunk]),
            )
        return (
            200,
            {
                "content-type": download_mime,
                "content-length": str(len(blob)),
                "accept-ranges": "bytes",
            },
            _DriveBody([blob]),
        )

    monkeypatch.setattr(cloud_providers, "http_bytes", _bytes)


def _complete_cloud_video(client, admin, monkeypatch) -> str:
    """Create a Google-backed video row through the real cloud routes."""
    _connect(client, admin, GOOGLE, "me@gmail.com", monkeypatch)
    _fake_refresh_and_upload(monkeypatch)
    start = client.post(
        f"{API}/cloud/uploads", json={"backend": GOOGLE, "filename": "clip.mp4"}, headers=admin
    ).json()
    resp = client.post(
        f"{API}/cloud/uploads/{start['video_id']}/complete",
        json={"provider_file_id": "file-1"},
        headers=admin,
    )
    assert resp.status_code == 200, resp.text
    return start["video_id"]


def test_cloud_range_request_returns_206_with_the_provider_content_range(
    client, drive_config, monkeypatch
):
    blob = bytes(range(256)) * 168 + b"x" * 40  # 43048 bytes
    admin = _admin(client)
    _fake_provider_http(monkeypatch, blob)
    video_id = _complete_cloud_video(client, admin, monkeypatch)

    resp = client.get(
        f"{API}/videos/{video_id}/media",
        headers={**admin, "Range": "bytes=0-99"},
    )

    assert resp.status_code == 206, resp.text
    assert resp.headers["content-range"] == f"bytes 0-99/{len(blob)}"
    # The ranged length, not the whole file's size.
    assert resp.headers["content-length"] == "100"
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.content == blob[:100]


def test_cloud_full_request_returns_200_with_the_total_length(
    client, drive_config, monkeypatch
):
    blob = bytes(range(256)) * 168 + b"x" * 40
    admin = _admin(client)
    # The probe is only an optimisation: even when it cannot report a size,
    # the download response's own Content-Length must reach the client.
    _fake_provider_http(monkeypatch, blob, probe_has_size=False)
    video_id = _complete_cloud_video(client, admin, monkeypatch)

    resp = client.get(f"{API}/videos/{video_id}/media", headers=admin)

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-length"] == str(len(blob))
    assert resp.headers["accept-ranges"] == "bytes"
    assert "content-range" not in resp.headers
    assert resp.content == blob


def test_cloud_content_type_comes_from_the_download_not_the_probe(
    client, drive_config, monkeypatch
):
    blob = b"\x00\x00\x00\x18ftypqt  " + b"q" * 64
    admin = _admin(client)
    # The probe and the download disagree on purpose: the download wins.
    _fake_provider_http(
        monkeypatch, blob, probe_mime="video/mp4", download_mime="video/quicktime"
    )
    video_id = _complete_cloud_video(client, admin, monkeypatch)

    resp = client.get(f"{API}/videos/{video_id}/media", headers=admin)

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "video/quicktime"


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
