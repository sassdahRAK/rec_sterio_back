"""Bot users: consent records, per-user cookies storage and the ingest gate.

The /bot-users routes require BOTH the bot service token and the user's own
Telegram id, so one user can never act on another's cookies. The consent
gate in POST /videos enforces that a gated (TikTok/Facebook) download runs
under the requesting user's own session.
"""
from __future__ import annotations

import hashlib
import os

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import BotUser, Job
from app.routers.bot_users import CONSENT_VERSION
from app.services import storage
from tests.helpers import telegram_user_headers

UID = 12345
TIKTOK_URL = "https://www.tiktok.com/@someplace/video/7000000000000000001"
YOUTUBE_URL = "https://www.youtube.com/watch?v=recsterio-botusers-001"

COOKIES_TXT = (
    "# Netscape HTTP Cookie File\n"
    "# https://curl.se/docs/http-cookies.html\n"
    "\n"
    ".tiktok.com\tTRUE\t/\tTRUE\t1735689600\tsessionid\tabc123def456\n"
    ".tiktok.com\tTRUE\t/\tTRUE\t1735689600\ttt-target-idc\tnone\n"
)


def _consent(client, uid: int = UID) -> None:
    resp = client.post(
        f"/api/v1/bot-users/{uid}/consent",
        json={"consent_version": CONSENT_VERSION},
        headers=telegram_user_headers(uid),
    )
    assert resp.status_code == 200, resp.text


def _row(uid: int = UID) -> BotUser | None:
    with SessionLocal() as db:
        return db.execute(
            select(BotUser).where(BotUser.telegram_user_id == uid)
        ).scalar_one_or_none()


# ------------------------------------------------------------ credential gate


def test_get_unknown_user_is_404(client):
    resp = client.get(f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_non_numeric_id_is_a_validation_error(client):
    resp = client.get(
        "/api/v1/bot-users/not-a-number", headers=telegram_user_headers(UID)
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_failed"


def test_bad_bearer_token_is_401(client):
    resp = client.get(
        f"/api/v1/bot-users/{UID}",
        headers={
            "Authorization": "Bearer not-the-service-token",
            "X-Requested-By": f"telegram:{UID}",
        },
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "not_authenticated"


def test_missing_bearer_token_is_401(client):
    resp = client.get(
        f"/api/v1/bot-users/{UID}", headers={"X-Requested-By": f"telegram:{UID}"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "not_authenticated"


def test_another_users_identity_is_403(client):
    # The service token alone is the bot's identity; acting on user 12345's
    # record while claiming to be user 999 must be refused.
    resp = client.get(
        f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(999)
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


def test_bot_service_token_without_a_user_header_is_403(client):
    resp = client.get(
        f"/api/v1/bot-users/{UID}",
        headers={"Authorization": f"Bearer {settings.bot_api_token}"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


# ------------------------------------------------------------------- consent


def test_consent_records_version_and_timestamp(client):
    resp = client.post(
        f"/api/v1/bot-users/{UID}/consent",
        json={"consent_version": CONSENT_VERSION},
        headers=telegram_user_headers(UID),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["telegram_id"] == UID
    assert body["consented"] is True
    assert body["consent_version"] == CONSENT_VERSION
    assert isinstance(body["consented_at"], str)
    assert body["cookies_present"] is False

    got = client.get(f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID))
    assert got.status_code == 200
    assert got.json()["consented"] is True
    assert got.json()["consent_version"] == CONSENT_VERSION


def test_consent_version_mismatch_is_rejected(client):
    resp = client.post(
        f"/api/v1/bot-users/{UID}/consent",
        json={"consent_version": "2000-01-01.0"},
        headers=telegram_user_headers(UID),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "version_mismatch"
    # Nothing was recorded.
    again = client.get(f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID))
    assert again.status_code == 404


# -------------------------------------------------------------------- cookies


def test_cookies_upload_happy_path(client):
    _consent(client)
    resp = client.post(
        f"/api/v1/bot-users/{UID}/cookies",
        files={"file": ("cookies.txt", COOKIES_TXT.encode(), "text/plain")},
        headers=telegram_user_headers(UID),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cookies_present"] is True
    assert isinstance(body["cookies_updated_at"], str)

    path = storage.cookie_file(UID)
    assert path.exists()
    # A live session: only its owner may read it.
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert path.read_text() == COOKIES_TXT

    row = _row()
    assert row is not None
    assert row.cookies_sha256 == hashlib.sha256(COOKIES_TXT.encode()).hexdigest()
    assert row.cookies_updated_at is not None

    got = client.get(f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID))
    assert got.status_code == 200
    assert got.json()["cookies_present"] is True


def test_cookies_upload_rejects_non_netscape_content(client):
    _consent(client)
    for content in (
        b"this is just plain text, not a cookies export",
        b"# Netscape HTTP Cookie File\n# header but no cookie lines\n",
    ):
        resp = client.post(
            f"/api/v1/bot-users/{UID}/cookies",
            files={"file": ("cookies.txt", content, "text/plain")},
            headers=telegram_user_headers(UID),
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "invalid_cookies_file"
    assert not storage.cookie_file(UID).exists()


def test_cookies_upload_rejects_oversize_files(client, monkeypatch):
    monkeypatch.setattr(settings, "max_cookies_kb", 1)
    padded = COOKIES_TXT.replace("abc123def456", "x" * 2000)
    resp = client.post(
        f"/api/v1/bot-users/{UID}/cookies",
        files={"file": ("cookies.txt", padded.encode(), "text/plain")},
        headers=telegram_user_headers(UID),
    )
    assert resp.status_code == 413, resp.text
    assert resp.json()["error"]["code"] == "payload_too_large"
    assert not storage.cookie_file(UID).exists()


def test_withdraw_deletes_the_cookies_file_and_the_row(client):
    _consent(client)
    upload = client.post(
        f"/api/v1/bot-users/{UID}/cookies",
        files={"file": ("cookies.txt", COOKIES_TXT.encode(), "text/plain")},
        headers=telegram_user_headers(UID),
    )
    assert upload.status_code == 200

    resp = client.delete(f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID))
    assert resp.status_code == 204
    assert not storage.cookie_file(UID).exists()
    assert _row() is None
    got = client.get(f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID))
    assert got.status_code == 404


def test_withdraw_is_idempotent(client):
    resp = client.delete(f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID))
    assert resp.status_code == 204
    again = client.delete(
        f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID)
    )
    assert again.status_code == 204


# ------------------------------------------------- the consent gate on /videos


def test_gate_blocks_a_tiktok_download_without_consent(client):
    resp = client.post(
        "/api/v1/videos",
        json={"source_url": TIKTOK_URL},
        headers=telegram_user_headers(999),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "consent_required"
    # The gate fires before the insert, so no video and no job may exist.
    with SessionLocal() as db:
        assert db.execute(select(Job)).scalars().all() == []


def test_gate_passes_after_consent_and_cookies(client):
    uid = 999
    _consent(client, uid)
    upload = client.post(
        f"/api/v1/bot-users/{uid}/cookies",
        files={"file": ("cookies.txt", COOKIES_TXT.encode(), "text/plain")},
        headers=telegram_user_headers(uid),
    )
    assert upload.status_code == 200

    resp = client.post(
        "/api/v1/videos",
        json={"source_url": TIKTOK_URL},
        headers=telegram_user_headers(uid),
    )
    assert resp.status_code == 201, resp.text
    video_id = resp.json()["id"]

    with SessionLocal() as db:
        job = (
            db.execute(
                select(Job).where(Job.video_id == video_id, Job.kind == "download")
            )
            .scalars()
            .one_or_none()
        )
        assert job is not None
        assert job.status == "queued"


def test_gate_ignores_admin_and_youtube_downloads(client, auth_headers):
    # The operator is never gated...
    resp = client.post(
        "/api/v1/videos", json={"source_url": TIKTOK_URL}, headers=auth_headers
    )
    assert resp.status_code == 201, resp.text
    # ...and neither is a Telegram user on an un-gated platform.
    resp = client.post(
        "/api/v1/videos",
        json={"source_url": YOUTUBE_URL},
        headers=telegram_user_headers(999),
    )
    assert resp.status_code == 201, resp.text


def test_dedupe_still_fires_before_the_gate(client, auth_headers):
    """A URL the caller can already see is returned, even before consent.

    Dedupe runs ahead of the cookie gate so re-pasting a known link never
    demands cookies for a download that will not happen. The scope is what the
    caller may read: a *public* unit is visible to any bot user, so the 409
    (with the existing id) still fires for an un-gated one. A private unit does
    not match — see `test_a_private_unit_is_not_disclosed_by_dedupe`.
    """
    folder = client.post(
        "/api/v1/folders",
        json={"name": "Public dedupe", "visibility": "public"},
        headers=auth_headers,
    )
    assert folder.status_code == 201, folder.text
    created = client.post(
        "/api/v1/videos",
        json={"source_url": TIKTOK_URL, "folder_id": folder.json()["id"]},
        headers=auth_headers,
    )
    assert created.status_code == 201

    resp = client.post(
        "/api/v1/videos",
        json={"source_url": TIKTOK_URL},
        headers=telegram_user_headers(999),  # never consented
    )
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["error"]["code"] == "conflict"
    assert body["video"]["id"] == created.json()["id"]


def test_a_private_unit_is_not_disclosed_by_dedupe(client, auth_headers):
    """Dedupe must not be an oracle over someone else's private folder.

    A bot user who never consented and holds no role on the folder gets the
    consent gate (403), not the private video's id.
    """
    folder = client.post(
        "/api/v1/folders",
        json={"name": "Private dedupe", "visibility": "private"},
        headers=auth_headers,
    )
    assert folder.status_code == 201, folder.text
    created = client.post(
        "/api/v1/videos",
        json={"source_url": TIKTOK_URL, "folder_id": folder.json()["id"]},
        headers=auth_headers,
    )
    assert created.status_code == 201

    resp = client.post(
        "/api/v1/videos",
        json={"source_url": TIKTOK_URL},
        headers=telegram_user_headers(1001),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "consent_required"