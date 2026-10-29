"""Security hardening packets SK1–SK3 plus the review fixes on top: proxy-aware
client addressing for the limiter, boundary-sanitized job/transcript errors,
deletion cleanup of what no cascade reaches, the capped redeem search, the
ffprobe upload gate and CORS credential posture.
"""
from __future__ import annotations

import collections
import hashlib
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy import select

import pytest

from app.config import ConfigError, Settings, validate_security_settings
from app.db import SessionLocal
from app.errors import AppError
from app.models import (
    AdminUser,
    DriveLink,
    Folder,
    FolderMember,
    FolderPasscode,
    Job,
    LinkCode,
    Merge,
    PasscodeGrant,
    ShareLink,
    Transcript,
    Video,
)
from app.routers.passcodes import CANDIDATE_LIMIT, new_code_hash, verify_code
from app.routers.shares import token_digest
from app.security import RateLimiter, assert_fetch_url_safe, rate_limiter, validate_source_url
from app.services import cloud_providers
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD
from tests.helpers import ensure_user, make_folder, media_bytes, upload_video, user_headers

API = "/api/v1"


def account_token(email: str, role: str) -> dict:
    from app.auth import create_access_token

    token, _ = create_access_token(email, role)
    return {"Authorization": f"Bearer {token}"}


def operator() -> dict:
    return account_token(ADMIN_EMAIL, "super_admin")


def new_account(db_session, email: str, role: str) -> AdminUser:
    user = AdminUser(email=email, role=role)
    db_session.add(user)
    db_session.commit()
    return user


# ------------------------------------------------------------------ SK1 boot guard

PLACEHOLDERS = {
    "SECRET_KEY": "change-me-to-a-long-random-string",
    "ADMIN_PASSWORD": "change-me",
    "BOT_API_TOKEN": "replace-me-with-a-shared-service-token",
}


def _placeholder_settings(monkeypatch) -> Settings:
    for env, value in PLACEHOLDERS.items():
        monkeypatch.setenv(env, value)
    monkeypatch.delenv("ALLOW_INSECURE", raising=False)
    return Settings()


def test_boot_guard_refuses_the_three_placeholders_and_names_them(monkeypatch):
    """A fresh checkout's defaults must not boot a real deployment; the error
    has to say WHICH vars are still placeholders."""
    s = _placeholder_settings(monkeypatch)
    with pytest.raises(ConfigError) as excinfo:
        validate_security_settings(s)
    message = str(excinfo.value)
    for env_name in ("SECRET_KEY", "ADMIN_PASSWORD", "BOT_API_TOKEN"):
        assert env_name in message, message


def test_boot_guard_refuses_each_placeholder_individually(monkeypatch):
    real = {env: f"real-{env.lower()}-value-for-tests" for env in PLACEHOLDERS}
    for offending in PLACEHOLDERS:
        for env, value in real.items():
            monkeypatch.setenv(env, value)
        monkeypatch.setenv(offending, PLACEHOLDERS[offending])
        monkeypatch.delenv("ALLOW_INSECURE", raising=False)
        s = Settings()
        with pytest.raises(ConfigError) as excinfo:
            validate_security_settings(s)
        assert offending in str(excinfo.value)


def test_boot_guard_permits_real_values(monkeypatch):
    monkeypatch.delenv("ALLOW_INSECURE", raising=False)
    for env, real in (
        ("SECRET_KEY", "real-random-secret-for-this-deployment"),
        ("ADMIN_PASSWORD", "a-real-operator-password"),
        ("BOT_API_TOKEN", "a-real-service-token"),
    ):
        monkeypatch.setenv(env, real)
    validate_security_settings(Settings())  # no raise


def test_allow_insecure_is_the_local_dev_escape(monkeypatch):
    s = _placeholder_settings(monkeypatch)
    monkeypatch.setenv("ALLOW_INSECURE", "1")
    validate_security_settings(s)  # no raise with placeholders


# ------------------------------------------------------------------ SK1 auth role semantics


def test_the_role_claim_is_a_hint_the_database_row_wins(client, caplog):
    """A hand-crafted (or stale) token claiming 'admin' for a 'user' row runs
    with the row's role — and the drift is logged as one line, no secrets."""
    import logging

    email = ensure_user("drift")
    stale = account_token(email, "admin")
    with caplog.at_level(logging.INFO, logger="recsterio.auth"):
        resp = client.get(f"{API}/auth/me", headers=stale)
    assert resp.status_code == 200
    assert resp.json()["role"] == "user"
    assert any("role_drift" in r.message for r in caplog.records), caplog.text


def test_a_deactivated_row_is_refused_on_the_jwt_path(client):
    """The re-read requires an active row: is_active=false kills the account's
    existing tokens on their next request."""
    email = ensure_user("inactive-check")
    headers = account_token(email, "user")
    assert client.get(f"{API}/auth/me", headers=headers).status_code == 200

    admin = operator()
    roster = client.get(f"{API}/users", headers=admin).json()["items"]
    user_id = next(u["id"] for u in roster if u["email"] == email)
    assert client.patch(
        f"{API}/users/{user_id}", json={"is_active": False}, headers=admin
    ).status_code == 200
    assert client.get(f"{API}/auth/me", headers=headers).status_code == 401


# ------------------------------------------------------------------ SK2 rate limiter


def test_limiter_store_stays_bounded_and_lru_evicts():
    """A 6000-key flood cannot grow the store past its cap; recent keys win."""
    limiter = RateLimiter()
    for i in range(RateLimiter.MAX_KEYS + 700):
        limiter.check(f"key-{i}", limit=5, window_seconds=3600)
    assert len(limiter._hits) <= RateLimiter.MAX_KEYS
    # The most recently touched key still has room in its own bucket.
    assert limiter.check(f"key-{RateLimiter.MAX_KEYS + 699}", 5, 3600)[0] is True


def test_limiter_sweeps_expired_entries():
    now_stale = time.monotonic() - 7200
    limiter = RateLimiter()
    limiter._max_window_seen = 3600.0
    for i in range(RateLimiter.MAX_KEYS + 5):
        limiter._hits[f"stale-{i}"] = collections.deque([now_stale])
    allowed, _ = limiter.check("fresh-key", limit=5, window_seconds=3600)
    assert allowed is True
    assert len(limiter._hits) <= RateLimiter.MAX_KEYS
    assert not any(k.startswith("stale-") for k in list(limiter._hits)[:50])


def test_redeem_passcode_is_rate_limited_with_retry_after(client, monkeypatch):
    folder = make_folder(client, operator())
    rate_limiter.reset()
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "public_redeem_rate_limit", 2)
    monkeypatch.setattr(app_settings, "public_redeem_rate_window", 60)
    try:
        wrong = {"code": "AAAA-BBBB", "folder_id": folder["id"]}
        first = client.post(f"{API}/public/passcodes/redeem", json=wrong)
        second = client.post(f"{API}/public/passcodes/redeem", json=wrong)
        third = client.post(f"{API}/public/passcodes/redeem", json=wrong)
        # Misses are 401s as documented — the throttle still counts them.
        assert first.status_code == 401
        assert second.status_code == 401
        assert third.status_code == 429
        body = third.json()
        assert body["error"]["code"] == "rate_limited"
        assert "message" in body["error"]
        assert third.headers.get("Retry-After", "").isdigit()
    finally:
        rate_limiter.reset()


def test_redeem_share_is_rate_limited_with_retry_after(client, monkeypatch):
    admin = operator()
    folder = make_folder(client, admin)
    share = client.post(
        f"{API}/folders/{folder['id']}/shares", json={}, headers=admin
    ).json()
    rate_limiter.reset()
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "public_redeem_rate_limit", 2)
    monkeypatch.setattr(app_settings, "public_redeem_rate_window", 60)
    try:
        missed = [
            client.post(f"{API}/public/shares/{guess}/redeem", json={})
            for guess in ("no-such-link-one", "no-such-link-two")
        ]
        # Guessing unknown links is 404 (dead links reveal nothing) — and a
        # guess attempt is still a redeem attempt.
        assert all(r.status_code == 404 for r in missed)
        third = client.post(f"{API}/public/shares/{share['token']}/redeem", json={})
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "rate_limited"
        assert third.headers.get("Retry-After", "").isdigit()
    finally:
        rate_limiter.reset()


# ------------------------------------------------------------------ SK2 passcode storage


def test_new_passcodes_are_bcrypt_and_legacy_rows_still_verify():
    """New codes hash with bcrypt; a stored unsalted-sha256 row still verifies.
    The verify helper keeps accepting both so lazy rehash can exist at all."""
    code = "ABCD-2345"
    bcrypt_hash = new_code_hash(code)
    assert bcrypt_hash.startswith("$2")
    assert verify_code(code, bcrypt_hash) is True
    assert verify_code("ABCD-2346", bcrypt_hash) is False

    legacy_hash = hashlib.sha256(code.encode()).hexdigest()
    assert verify_code(code, legacy_hash) is True
    assert verify_code("wrong-code", legacy_hash) is False


def test_a_bcrypt_passcode_redeems(client):
    admin = operator()
    folder = make_folder(client, admin)
    code = client.post(
        f"{API}/folders/{folder['id']}/passcodes", json={}, headers=admin
    ).json()["code"]
    resp = client.post(
        f"{API}/public/passcodes/redeem",
        json={"code": code, "folder_id": folder["id"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["access"] == "passcode"


def test_the_generated_code_is_returned_once(client):
    """Byte-identical contract behavior: the plaintext lives only in the 201."""
    admin = operator()
    folder = make_folder(client, admin)
    created = client.post(
        f"{API}/folders/{folder['id']}/passcodes", json={}, headers=admin
    ).json()
    assert created["code"]
    listed = client.get(f"{API}/folders/{folder['id']}/passcodes", headers=admin).json()
    returned = next(p for p in listed if p["id"] == created["id"])
    assert returned["code"] is None


def test_redeeming_a_legacy_passcode_converts_it_in_place(client):
    """A pre-hardening sha256 code keeps opening doors and comes back bcrypt:
    lazy rehash on successful use, no data migration anywhere."""
    admin = operator()
    folder = make_folder(client, admin)
    client.post(
        f"{API}/folders/{folder['id']}/passcodes",
        json={"code": "LEGACY-01"},
        headers=admin,
    )
    with SessionLocal() as db:
        row = db.execute(
            select(FolderPasscode).where(FolderPasscode.folder_id == folder["id"])
        ).scalar_one()
        row.code_hash = hashlib.sha256("LEGACY-01".encode()).hexdigest()
        db.commit()

    resp = client.post(
        f"{API}/public/passcodes/redeem",
        json={"code": "LEGACY-01", "folder_id": folder["id"]},
    )
    assert resp.status_code == 200, resp.text
    with SessionLocal() as db:
        row = db.execute(
            select(FolderPasscode).where(FolderPasscode.folder_id == folder["id"])
        ).scalar_one()
        assert row.code_hash.startswith("$2")


# ------------------------------------------------------------------ SK2 SSRF at fetch time


def test_fetch_guard_blocks_an_unresolvable_host(monkeypatch):
    """At fetch time an unresolvable hostname is refused (fail closed), where
    create-time validation passes it for the worker to fail on later.
    DNS is stubbed so the test itself stays hermetic."""

    def _unresolvable(host, *args, **kwargs):
        raise __import__("socket").gaierror("name or service not known")

    monkeypatch.setattr("app.security.socket.getaddrinfo", _unresolvable)
    with pytest.raises(AppError) as excinfo:
        assert_fetch_url_safe("https://no-such-host.example.org/media.mp4")
    assert excinfo.value.details[0]["issue"] == "blocked_host"


def test_fetch_guard_blocks_a_host_that_resolves_private(monkeypatch):
    import socket as socket_module

    def _fake_getaddrinfo(host, *args, **kwargs):
        # One public address and one internal one in the answer set: any
        # dangerous resolved address alone refuses the fetch.
        return [
            (socket_module.AF_INET, socket_module.SOCK_STREAM, 6, "", ("8.8.8.8", 0)),
            (socket_module.AF_INET, socket_module.SOCK_STREAM, 6, "", ("10.0.0.9", 0)),
        ]

    monkeypatch.setattr("app.security.socket.getaddrinfo", _fake_getaddrinfo)
    with pytest.raises(AppError) as excinfo:
        assert_fetch_url_safe("https://looks-public.example.org/video.mp4")
    assert excinfo.value.details[0]["issue"] == "blocked_host"


def test_create_time_validation_keeps_its_unresolvable_allowance():
    """Create-time validation stays best-effort on resolution (the worker
    would fail cleanly); the fetch-time guard is the one that fails closed."""
    url = "https://no-such-host.recsterio.invalid/media.mp4"
    assert validate_source_url(url) == url


def test_downloader_validates_the_url_before_the_fetch(monkeypatch, tmp_path):
    """The download step itself re-runs the guard — create-time validation
    happened long before the worker actually pulled bytes. A literal public
    IP keeps the test DNS-free; the spy replaces the resolver entirely."""
    from app.services import downloader
    from tests.test_downloader import _FakeYoutubeDLError, _clear_ytdlp_settings, _install_fake_ytdlp

    _clear_ytdlp_settings(monkeypatch)
    monkeypatch.setattr(downloader.settings, "media_root", str(tmp_path))

    seen: list[str] = []

    def _spy(url: str) -> str:
        seen.append(url)
        return url

    monkeypatch.setattr(downloader, "assert_fetch_url_safe", _spy)

    def _stop(self, url, download=True):
        raise _FakeYoutubeDLError("stop after construction")

    _install_fake_ytdlp(monkeypatch, extract_info=_stop)
    source = "https://93.184.216.34/video.mp4"  # a literal public address
    with pytest.raises(downloader.TransientDownloadError):
        downloader.download(source, uuid.uuid4())
    assert seen == [source]


def test_downloader_discards_a_redirect_that_lands_on_a_private_address(
    monkeypatch, tmp_path
):
    """Redirect hops happen inside yt-dlp; the final URL is re-checked after
    the fetch and the bytes are gone if the destination was internal. Both
    endpoints are literal addresses, so no DNS is touched."""
    from app.services import downloader
    from tests.test_downloader import _clear_ytdlp_settings, _install_fake_ytdlp

    _clear_ytdlp_settings(monkeypatch)
    monkeypatch.setattr(downloader.settings, "media_root", str(tmp_path))

    video_id = uuid.uuid4()
    outdir = downloader.storage.video_dir(video_id)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "media.mp4"
    dest.write_bytes(media_bytes())

    def _extract(self, url, download=True):
        return {
            "title": "harmless looking public video",
            "url": "http://169.254.169.254/latest/meta-data/",
            "requested_downloads": [{"filepath": str(dest)}],
        }

    _install_fake_ytdlp(monkeypatch, extract_info=_extract)
    with pytest.raises(downloader.DownloadError) as excinfo:
        downloader.download("https://93.184.216.34/video.mp4", video_id)
    assert "private" in str(excinfo.value)
    assert not dest.exists()


# ------------------------------------------------------------------ SK2 upload sniffing


def test_upload_accepts_a_real_container_header(client, auth_headers):
    resp = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.mp4", media_bytes(), "video/mp4")},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text


def test_upload_rejects_bytes_that_do_not_match_the_extension(client, auth_headers):
    """An HTML page renamed .mp4 must not reach ffmpeg (the common upload trap)."""
    resp = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.mp4", b"<html><body>not a video</body></html>", "video/mp4")},
        headers=auth_headers,
    )
    assert resp.status_code == 415
    body = resp.json()
    assert body["error"]["code"] == "unsupported_media_type"
    assert body["error"]["details"][0]["issue"] == "content_mismatch"


def test_upload_rejects_an_extension_and_container_mismatch(client, auth_headers):
    """Valid mp4 bytes named .webm: the container has to match the name too."""
    resp = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.webm", media_bytes(), "video/webm")},
        headers=auth_headers,
    )
    assert resp.status_code == 415
    assert resp.json()["error"]["details"][0]["issue"] == "content_mismatch"


# ------------------------------------------------------------------ SK2 dedupe redaction


def test_developer_private_collision_409_is_redacted(client):
    """A developer's global dedupe view must not disclose somebody else's
    private unit: the 409 carries no video object, id, title or path."""
    owner = ensure_user("private-owner")
    owner_headers = user_headers(owner)
    folder = make_folder(client, owner_headers, name="Private set")
    created = client.post(
        f"{API}/videos",
        json={
            "source_url": "https://example.org/private-clip-2",
            "folder_id": folder["id"],
            "title": "Private working clip",
        },
        headers=owner_headers,
    ).json()

    with SessionLocal() as db:
        dev_row = new_account(db, f"dev-{uuid.uuid4().hex[:8]}@test.local", "developer")
    resp = client.post(
        f"{API}/videos",
        json={"source_url": "https://example.org/private-clip-2"},
        headers=account_token(dev_row.email, "developer"),
    )
    assert resp.status_code == 409
    body = resp.json()
    # The generic conflict: no video object, no id, no title, no paths.
    assert "video" not in body
    assert "details" not in body["error"]
    assert created["title"] not in (body["error"]["message"] or "")
    assert str(created["id"]) not in (body["error"]["message"] or "")

    # Contrast: the OWNER re-submitting still gets the full envelope to reuse.
    resp_owner = client.post(
        f"{API}/videos",
        json={"source_url": "https://example.org/private-clip-2"},
        headers=owner_headers,
    )
    assert resp_owner.status_code == 409
    assert "video" in resp_owner.json()
    assert resp_owner.json()["error"]["details"][0]["video_id"] == created["id"]


def test_developer_public_collision_409_still_carries_the_video(client):
    """Idempotency keeps its useful half: a public unit's collision is echoed."""
    admin = operator()
    folder = make_folder(client, admin, visibility="public", name="Public set")
    created = client.post(
        f"{API}/videos",
        json={"source_url": "https://example.org/public-clip-1", "folder_id": folder["id"]},
        headers=admin,
    ).json()

    with SessionLocal() as db:
        dev_row = new_account(db, f"dev-{uuid.uuid4().hex[:8]}@test.local", "developer")
    resp = client.post(
        f"{API}/videos",
        json={"source_url": "https://example.org/public-clip-1"},
        headers=account_token(dev_row.email, "developer"),
    )
    assert resp.status_code == 409
    body = resp.json()
    assert "video" in body
    assert body["video"]["id"] == created["id"]


def test_upload_dedupe_409_is_redacted_the_same_way(client):
    """Same rule for the checksum dedupe on /videos/upload: identical bytes."""
    owner = ensure_user("dup-uploader")
    owner_headers = user_headers(owner)
    folder = make_folder(client, owner_headers, name="Private dup set")
    payload = media_bytes()
    first = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.mp4", payload, "video/mp4")},
        data={"folder_id": str(folder["id"]), "title": "Owned clip"},
        headers=owner_headers,
    )
    assert first.status_code == 201, first.text

    with SessionLocal() as db:
        dev_row = new_account(db, f"dev-{uuid.uuid4().hex[:8]}@test.local", "developer")
    resp = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.mp4", payload, "video/mp4")},
        headers=account_token(dev_row.email, "developer"),
    )
    assert resp.status_code == 409
    body = resp.json()
    assert "video" not in body
    assert "details" not in body["error"]
    assert "Owned clip" not in (body["error"]["message"] or "")


def test_patch_transcript_flips_the_badge_without_a_new_version(client):
    """Contract §PATCH: the cleaned FLAG flips in place on the current version
    and no new version is created — history is never rewritten by PATCH."""
    admin = operator()
    video = upload_video(client, admin)
    client.put(
        f"{API}/videos/{video['id']}/transcript",
        json={"content": "manual line one"},
        headers=admin,
    )
    resp = client.patch(
        f"{API}/videos/{video['id']}/transcript",
        json={"cleaned": False},
        headers=admin,
    )
    assert resp.status_code == 200
    assert resp.json()["cleaned"] is False
    versions = client.get(f"{API}/videos/{video['id']}/transcripts", headers=admin).json()
    assert len(versions["items"]) == 1


# ------------------------------------------------------------------ SK3 response leaks


def test_video_error_is_sane_at_the_boundary(client, auth_headers):
    """A failed step's stored error may carry paths and stderr lines; what the
    API says back is one line, path-free, and bounded."""
    video_id = client.post(
        f"{API}/videos", json={"source_url": "https://example.org/boom-clip"}, headers=auth_headers
    ).json()["id"]
    with SessionLocal() as db:
        video = db.get(Video, video_id)
        video.error = (
            "ffmpeg failed to extract audio: /Users/mac/server/storage/media/"
            f"{video_id}/media.mp4: Invalid data found when processing input\n"
            "[mov,mp4 @ 0x7fa31c004380] stream 0: offset changed\n"
            "Error splitting the input into NAL units."
        )
        db.commit()
    detail = client.get(f"{API}/videos/{video_id}", headers=auth_headers).json()
    error = detail["error"]
    assert "\n" not in error
    assert "/Users/" not in error
    assert not error.startswith("/")
    assert "[path]" in error  # where the path used to be
    assert len(error) <= 300


def test_video_error_keeps_short_human_reasons(client, auth_headers):
    video_id = client.post(
        f"{API}/videos", json={"source_url": "https://example.org/human-clip"}, headers=auth_headers
    ).json()["id"]
    with SessionLocal() as db:
        db.get(Video, video_id).error = "the site is rate-limiting this server"
        db.commit()
    detail = client.get(f"{API}/videos/{video_id}", headers=auth_headers).json()
    assert detail["error"] == "the site is rate-limiting this server"


def test_video_out_never_carries_local_path(client, auth_headers):
    """Dropped from the schema entirely: an upload sets the column, no
    response ever carries it (list, detail, everywhere)."""
    video = upload_video(client, auth_headers)
    detail = client.get(f"{API}/videos/{video['id']}", headers=auth_headers)
    assert detail.status_code == 200
    assert "local_path" not in detail.json()
    listing = client.get(f"{API}/videos", headers=auth_headers).json()
    assert all("local_path" not in row for row in listing["items"])


# ------------------------------------------------------------------ SK3 share tokens at rest


def test_share_token_stored_as_digest_and_redeemable(client):
    """201 shows the plaintext once; the row stores only its sha256 digest."""
    admin = operator()
    folder = make_folder(client, admin)
    share = client.post(
        f"{API}/folders/{folder['id']}/shares", json={}, headers=admin
    ).json()
    with SessionLocal() as db:
        row = db.get(ShareLink, share["id"])
        assert row.token == token_digest(share["token"])
        assert row.token != share["token"]

    resp = client.post(f"{API}/public/shares/{share['token']}/redeem", json={})
    assert resp.status_code == 200, resp.text


def test_legacy_plaintext_share_token_is_rehash_stored_on_successful_use(client):
    """Rows written before the switch keep working and convert themselves."""
    admin = operator()
    folder = make_folder(client, admin)
    plain = "pre-hardening-plain-token"
    with SessionLocal() as db:
        db.add(
            ShareLink(
                folder_id=folder["id"], token=plain, role="viewer",
                created_by=f"admin:{ADMIN_EMAIL}",
            )
        )
        db.commit()

    resp = client.post(f"{API}/public/shares/{plain}/redeem", json={})
    assert resp.status_code == 200, resp.text
    with SessionLocal() as db:
        row = db.execute(select(ShareLink).where(ShareLink.folder_id == folder["id"])).scalar_one()
        assert row.token == token_digest(plain)


def test_an_unknown_share_token_is_still_the_same_answer(client):
    resp = client.post(f"{API}/public/shares/never-existed-token/redeem", json={})
    assert resp.status_code == 404


# ------------------------------------------------------------------ SK3 drive tokens (AESGCM)


_REAL_KEY = (b"x" * 32)


def _key_settings(monkeypatch) -> str:
    import base64

    key = base64.urlsafe_b64encode(_REAL_KEY).decode()
    monkeypatch.setattr(cloud_providers.settings, "cloud_token_key", key)
    return key


def test_aesgcm_roundtrip_and_envelope_marker(monkeypatch):
    _key_settings(monkeypatch)
    stored = cloud_providers.encrypt_token("refresh-token-abc")
    assert stored.startswith("enc1.")
    assert stored != "refresh-token-abc"
    assert "refresh-token-abc" not in stored
    assert cloud_providers.decrypt_token(stored) == "refresh-token-abc"


def test_encrypt_refuses_when_the_key_is_unset(monkeypatch):
    monkeypatch.delenv("ALLOW_INSECURE", raising=False)
    monkeypatch.setattr(cloud_providers.settings, "cloud_token_key", None)
    with pytest.raises(AppError) as excinfo:
        cloud_providers.encrypt_token("refresh-token-abc")
    assert excinfo.value.code == "cloud_token_key_required"
    assert "CLOUD_TOKEN_KEY" in excinfo.value.message


def test_allow_insecure_lets_local_dev_store_plaintext(monkeypatch):
    import os

    monkeypatch.setenv("ALLOW_INSECURE", "1")
    monkeypatch.setattr(cloud_providers.settings, "cloud_token_key", None)
    assert cloud_providers.encrypt_token("refresh-token-abc") == "refresh-token-abc"
    assert cloud_providers.decrypt_token("refresh-token-abc") == "refresh-token-abc"


def test_decrypt_refuses_a_raw_plaintext_row_and_a_corrupt_envelope(monkeypatch):
    """Key now configured, value written pre-key (or tampered): the token is
    never silently handed to a provider — the drive requires re-auth."""
    monkeypatch.delenv("ALLOW_INSECURE", raising=False)
    _key_settings(monkeypatch)
    for unusable in ("1//raw-google-refresh-token", "enc1.notreallybase64!!"):
        with pytest.raises(AppError) as excinfo:
            cloud_providers.decrypt_token(unusable)
        assert excinfo.value.code == "cloud_auth_rejected"


def test_a_tampered_aesgcm_envelope_is_refused(monkeypatch):
    _key_settings(monkeypatch)
    stored = cloud_providers.encrypt_token("refresh-token-abc")
    # Flip one ciphertext byte (after the 12-byte nonce, inside the b64 body).
    body = stored[len("enc1."):]
    tampered = body[:-4] + ("A" if body[-4] != "A" else "B") + body[-3:]
    with pytest.raises(AppError) as excinfo:
        cloud_providers.decrypt_token("enc1." + tampered)
    assert excinfo.value.code == "cloud_auth_rejected"


def test_legacy_envelope_rows_still_decrypt(monkeypatch):
    """Rows from the previous (still-signed) envelope keep reading."""
    import base64
    import hashlib
    import hmac
    import os
    import struct

    _key_settings(monkeypatch)
    key = base64.urlsafe_b64decode(
        cloud_providers.settings.cloud_token_key + "=="
    )
    aes_key = hashlib.sha256(b"aes:" + key).digest()
    mac_key = hashlib.sha256(b"mac:" + key).digest()
    nonce = os.urandom(12)
    data = b"refresh-token-legacy"
    keystream = b""
    counter = 0
    while len(keystream) < len(data):
        keystream += hashlib.sha256(aes_key + nonce + struct.pack(">Q", counter)).digest()
        counter += 1
    ciphertext = bytes(a ^ b for a, b in zip(data, keystream[: len(data)]))
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()[:16]
    stored = base64.urlsafe_b64encode(nonce + ciphertext + tag).decode()
    assert cloud_providers.decrypt_token(stored) == "refresh-token-legacy"


# ------------------------------------------------------------------ SK3 status not editor-writable


def test_patch_video_ignores_a_status_override(client, auth_headers):
    """The processing lifecycle belongs to the queue/runner; an editor cannot
    poke a row's status through the generic PATCH."""
    video_id = client.post(
        f"{API}/videos", json={"source_url": "https://example.org/no-status-change"},
        headers=auth_headers,
    ).json()["id"]
    resp = client.patch(
        f"{API}/videos/{video_id}",
        json={"status": "transcribed", "title": "still pending"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert body["title"] == "still pending"


# ------------------------------------------------------------------ SK3 DELETE /users/me


def _self_delete_flow(client):
    """A user with one folder, one video in it, and a comment on their folder."""
    email = ensure_user("selfdelete")
    headers = user_headers(email)
    folder = make_folder(client, headers, name="Mine to hand over")
    video = upload_video(client, headers, folder_id=folder["id"], title="Stays put")
    token_headers = account_token(email, "user")
    client.post(f"{API}/folders/{folder['id']}/comments", json={"body": "note"}, headers=headers)
    return email, headers, folder, video, token_headers


def test_delete_me_requires_confirmation_when_folders_are_owned(client):
    email, headers, folder, video, _ = _self_delete_flow(client)
    resp = client.delete(f"{API}/users/me", headers=headers)
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "confirmation_required"
    # Nothing changed by refusing.
    with SessionLocal() as db:
        row = __import__("sqlalchemy").select  # noqa: F841
        user = db.execute(
            select(AdminUser).where(AdminUser.email == email)
        ).scalar_one_or_none()
        assert user is not None
        assert db.get(Folder, folder["id"]) is not None


def test_delete_me_anonymizes_revokes_and_keeps_videos(client):
    email, headers, folder, video, token_headers = _self_delete_flow(client)
    resp = client.request("DELETE", f"{API}/users/me", json={"confirm": True}, headers=headers)
    assert resp.status_code == 204
    with SessionLocal() as db:
        # The old identity is not reachable by its email...
        row = db.execute(select(AdminUser).where(AdminUser.email == email)).scalar_one_or_none()
        assert row is None
        # ...but the row itself stays, anonymized to a marker (attribution history).
        accounts = db.execute(select(AdminUser)).scalars().all()
        markers = [a for a in accounts if a.email.startswith("deleted:")]
        assert len(markers) == 1
        marker = markers[0]
        assert marker.name.startswith("deleted:")
        assert marker.password_hash is None
        assert marker.avatar_url is None
        assert marker.google_sub is None
        assert marker.telegram_user_id is None
        # Folders went to the operator; the dataset videos inside are kept.
        transferred = db.execute(select(Folder)).scalars().all()
        assert len(transferred) == 1
        assert str(transferred[0].owner_id) == str(
            db.execute(
                select(AdminUser).where(AdminUser.email == ADMIN_EMAIL)
            ).scalar_one().id
        )
        kept = db.execute(select(Video)).scalars().all()
        assert len(kept) == 1
        assert str(kept[0].id) == str(video["id"])

    # Any token the account ever held is dead: the sub no longer resolves.
    assert client.get(f"{API}/auth/me", headers=token_headers).status_code == 401
    # So is the account (an identity cannot be re-created by re-signing in).
    assert client.delete(f"{API}/users/me", headers=token_headers).status_code == 401


def test_delete_me_refuses_super_admin_and_the_bot(client):
    # The super admin is the deployment's management surface.
    assert client.delete(f"{API}/users/me", headers=operator()).status_code == 422
    # The bot service token is not an account row at all.
    bot_headers = {
        "Authorization": "Bearer test-bot-service-token",
        "X-Requested-By": "telegram:bot",
    }
    resp = client.delete(f"{API}/users/me", headers=bot_headers)
    assert resp.status_code == 403


def test_delete_me_for_a_folderless_account_needs_no_confirm(client):
    email = ensure_user("light-delete")
    headers = user_headers(email)
    resp = client.delete(f"{API}/users/me", headers=headers)
    assert resp.status_code == 204


def _admin_login(client) -> dict:
    resp = client.post(
        f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def test_plain_login_is_untouched(client):
    """Sanity: the packets changed nothing about credential sign-in."""
    assert client.post(
        f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    ).status_code == 200
    assert (
        client.post(
            f"{API}/auth/login",
            json={"email": ADMIN_EMAIL, "password": "definitely-wrong"},
        ).status_code
        == 401
    )


# ------------------------------------------------------------------ review fixes: proxy trust


def test_xff_header_rotation_cannot_mint_fresh_redeem_buckets(client, monkeypatch):
    """Default policy trusts no client-supplied header: rotating
    `X-Forwarded-For` must not reset the redeem throttle — every miss lands in
    the same per-peer bucket, so the 429 stays put."""
    folder = make_folder(client, operator())
    from app.config import settings as app_settings

    rate_limiter.reset()
    monkeypatch.setattr(app_settings, "public_redeem_rate_limit", 2)
    monkeypatch.setattr(app_settings, "public_redeem_rate_window", 60)
    try:
        wrong = {"code": "AAAA-BBBB", "folder_id": folder["id"]}
        first = client.post(
            f"{API}/public/passcodes/redeem", json=wrong,
            headers={"X-Forwarded-For": "203.0.113.7"},
        )
        second = client.post(
            f"{API}/public/passcodes/redeem", json=wrong,
            headers={"X-Forwarded-For": "198.51.100.9"},
        )
        third = client.post(
            f"{API}/public/passcodes/redeem", json=wrong,
            headers={"X-Forwarded-For": "192.0.2.11"},
        )
        assert (first.status_code, second.status_code) == (401, 401)
        assert third.status_code == 429, third.text
        assert third.json()["error"]["code"] == "rate_limited"
    finally:
        rate_limiter.reset()


def test_login_throttle_also_survives_xff_rotation(client, monkeypatch):
    """Same policy at the login throttle (the other credential-guess surface)."""
    from app.config import settings as app_settings

    rate_limiter.reset()
    monkeypatch.setattr(app_settings, "login_rate_limit", 2)
    monkeypatch.setattr(app_settings, "login_rate_window", 60)
    try:
        for i, ip in enumerate(("203.0.113.20", "198.51.100.21", "192.0.2.22")):
            resp = client.post(
                f"{API}/auth/login",
                json={"email": "nobody@test.invalid", "password": "wrong"},
                headers={"X-Forwarded-For": ip},
            )
            assert resp.status_code == (401 if i < 2 else 429), resp.text
    finally:
        rate_limiter.reset()


def test_trust_proxy_headers_keys_on_the_last_forwarded_hop(client, monkeypatch):
    """Behind the same-host Caddy (`TRUST_PROXY_HEADERS=true`) the real client
    is the LAST hop — the one the proxy wrote. A different client gets its own
    bucket, and a caller cannot escape into a fresh one by claiming extra
    earlier hops, however the header rotates."""
    folder = make_folder(client, operator())
    from app.config import settings as app_settings

    rate_limiter.reset()
    monkeypatch.setattr(app_settings, "trust_proxy_headers", True)
    monkeypatch.setattr(app_settings, "public_redeem_rate_limit", 2)
    monkeypatch.setattr(app_settings, "public_redeem_rate_window", 60)
    try:
        wrong = {"code": "AAAA-BBBB", "folder_id": folder["id"]}
        path = f"{API}/public/passcodes/redeem"
        xff = {"X-Forwarded-For": "203.0.113.50"}
        first = client.post(path, json=wrong, headers=xff)
        second = client.post(path, json=wrong, headers=xff)
        assert (first.status_code, second.status_code) == (401, 401)
        third = client.post(path, json=wrong, headers=xff)  # bucket of one client fills
        assert third.status_code == 429, third.text
        # A genuinely different client is a different bucket.
        fresh = client.post(path, json=wrong, headers={"X-Forwarded-For": "198.51.100.200"})
        assert fresh.status_code == 401
        # Claims of earlier hops do not move the key: "1.2.3.4, 203.0.113.50"
        # keys on 203.0.113.50 (full bucket), not on the freshly minted "1.2.3.4".
        spoofed = client.post(
            path, json=wrong, headers={"X-Forwarded-For": "1.2.3.4, 203.0.113.50"}
        )
        assert spoofed.status_code == 429
    finally:
        rate_limiter.reset()


# ------------------------------------------------------------------ review fixes: more boundary sanitization


def test_job_and_transcript_errors_are_sanitized_too(client, auth_headers):
    """`GET /videos/{id}` embeds jobs and transcripts (`VideoDetailOut`): their
    error fields must obey the same one-line, path-free boundary rule."""
    video_id = client.post(
        f"{API}/videos",
        json={"source_url": "https://example.org/embedded-error-clip"},
        headers=auth_headers,
    ).json()["id"]
    with SessionLocal() as db:
        db.add(
            Job(
                kind="extract_audio",
                video_id=uuid.UUID(video_id),
                status="dead",
                attempts=3,
                last_error=(
                    "ffmpeg extraction failed: /var/lib/recsterio/storage/media/"
                    f"x/media.wav: No such file or directory\n"
                    '  File "/opt/recsterio/app/worker/runner.py", line 88, in run\n'
                    "  step(extract_audio)"
                ),
            )
        )
        db.add(
            Transcript(
                video_id=uuid.UUID(video_id),
                version=1,
                source="whisper",
                status="failed",
                content="",
                error=(
                    "RuntimeError: model path /Users/mac/.cache/whisper/small/en does not exist\n"
                    "during load"
                ),
            )
        )
        db.commit()

    detail = client.get(f"{API}/videos/{video_id}", headers=auth_headers).json()
    job_error = next(j for j in detail["jobs"] if j["status"] == "dead")["last_error"]
    transcript_error = next(t for t in detail["transcripts"])["error"]
    for field in (job_error, transcript_error):
        assert "\n" not in field
        assert "/opt/" not in field and "/Users/" not in field and "/var/lib" not in field
        assert "[path]" in field
        assert len(field) <= 300


# ------------------------------------------------------------------ review fixes: deletion cleanup


def test_delete_me_deletes_drive_links_and_passcode_grants(client):
    """The row is anonymized, never deleted — no FK cascade can fire, so the
    deletion itself clears, in one transaction: the connected cloud links (a
    live OAuth refresh token + provider account email), outstanding Telegram
    link codes, folder memberships, and the device grants of the passcodes
    this account minted (those passcodes are revoked: none keeps opening
    doors)."""
    email = ensure_user("residual-clean")
    headers = user_headers(email)
    folder = make_folder(client, headers, name="Residual set")
    client.post(
        f"{API}/folders/{folder['id']}/passcodes",
        json={"code": "RESID-1234", "max_devices": 2},  # a device cap makes redemptions write grants
        headers=headers,
    )
    redeemed = client.post(
        f"{API}/public/passcodes/redeem",
        json={"code": "RESID-1234", "folder_id": folder["id"], "device_key": "visitor-device-9"},
    )
    assert redeemed.status_code == 200, redeemed.text
    with SessionLocal() as db:
        user = db.execute(select(AdminUser).where(AdminUser.email == email)).scalar_one()
        db.add(
            DriveLink(
                user_id=user.id,
                provider="google",
                account_email="hers@gmail.com",
                refresh_token="1//live-oauth-refresh-token",
            )
        )
        db.add(
            LinkCode(
                code_hash=hashlib.sha256(b"link-code-plaintext").hexdigest(),
                admin_user_id=user.id,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            )
        )
        db.add(FolderMember(folder_id=folder["id"], user_id=user.id, role="viewer"))
        db.commit()
        user_id = user.id

    resp = client.request("DELETE", f"{API}/users/me", json={"confirm": True}, headers=headers)
    assert resp.status_code == 204, resp.text
    with SessionLocal() as db:
        assert db.execute(
            select(DriveLink).where(DriveLink.user_id == user_id)
        ).scalars().all() == []
        assert db.execute(select(PasscodeGrant)).scalars().all() == []
        assert db.execute(
            select(LinkCode).where(LinkCode.admin_user_id == user_id)
        ).scalars().all() == []
        assert db.execute(
            select(FolderMember).where(FolderMember.user_id == user_id)
        ).scalars().all() == []
        minted = db.execute(
            select(FolderPasscode).where(FolderPasscode.folder_id == folder["id"])
        ).scalar_one()
        assert minted.revoked_at is not None
    # ...and the door the deleted identity minted no longer opens.
    assert client.post(
        f"{API}/public/passcodes/redeem", json={"code": "RESID-1234", "folder_id": folder["id"]}
    ).status_code == 401


def test_delete_me_does_not_touch_other_peoples_passcodes(client):
    """The cleanup is scoped: passcodes the deleted account did NOT mint stay
    exactly as they were (their grants included)."""
    kept = make_folder(client, operator(), name="Kept set")
    client.post(
        f"{API}/folders/{kept['id']}/passcodes",
        json={"code": "KEEPER-00", "max_devices": 3},
        headers=operator(),
    )
    assert client.post(
        f"{API}/public/passcodes/redeem",
        json={"code": "KEEPER-00", "folder_id": kept["id"], "device_key": "a-device"},
    ).status_code == 200
    with SessionLocal() as db:
        assert len(db.execute(select(PasscodeGrant)).scalars().all()) == 1
    email = ensure_user("bystander")
    resp = client.request(
        "DELETE", f"{API}/users/me", json={"confirm": True}, headers=user_headers(email)
    )
    assert resp.status_code == 204
    with SessionLocal() as db:
        grants = db.execute(select(PasscodeGrant)).scalars().all()
        assert len(grants) == 1  # the bystander minted nothing: the grant stays
        minted = db.execute(
            select(FolderPasscode).where(FolderPasscode.folder_id == kept["id"])
        ).scalar_one()
        assert minted.revoked_at is None


# ------------------------------------------------------------------ review fixes: capped redeem search


def test_no_target_redeem_is_capped_to_the_newest_live_candidates(client):
    """A wrong guess must not pay one bcrypt verify per live row: the
    no-target search runs newest-first with a SQL LIMIT. A live passcode
    beyond the newest `CANDIDATE_LIMIT` window stops redeeming — the
    documented residual surface."""
    folder = make_folder(client, operator())
    old_code, recent_code = "TOOOLD-001", "RECENT-002"
    actor = f"super_admin:{ADMIN_EMAIL}"
    with SessionLocal() as db:
        base = datetime.now(timezone.utc) - timedelta(hours=1)
        db.add(
            FolderPasscode(
                folder_id=folder["id"],
                code_hash=hashlib.sha256(old_code.encode()).hexdigest(),
                created_by=actor,
                created_at=base,  # the oldest row
            )
        )
        for i in range(CANDIDATE_LIMIT - 1):  # cheap legacy digests as filler
            db.add(
                FolderPasscode(
                    folder_id=folder["id"],
                    code_hash=hashlib.sha256(f"FILLER-{i:04d}".encode()).hexdigest(),
                    created_by=actor,
                    created_at=base + timedelta(seconds=i + 1),
                )
            )
        db.add(
            FolderPasscode(  # the newest row: well inside the window
                folder_id=folder["id"],
                code_hash=hashlib.sha256(recent_code.encode()).hexdigest(),
                created_by=actor,
                created_at=base + timedelta(hours=2),
            )
        )
        db.commit()

    resp = client.post(f"{API}/public/passcodes/redeem", json={"code": old_code})
    assert resp.status_code == 401  # capped: the oldest live code is not verified
    resp = client.post(f"{API}/public/passcodes/redeem", json={"code": recent_code})
    assert resp.status_code == 200, resp.text  # control: in-window codes open the door


# ------------------------------------------------------------------ review fixes: untested claims


def test_cors_preflight_never_allows_credentials(client):
    """Bearer auth cannot be cookie-backed, so allow_credentials stays False —
    checked both in the middleware configuration and on the wire."""
    from fastapi.middleware.cors import CORSMiddleware
    from app.config import settings
    from app.main import app as fastapi_app

    configured = [mw for mw in fastapi_app.user_middleware if mw.cls is CORSMiddleware]
    assert configured, "CORS middleware missing from the app"
    assert configured[0].kwargs.get("allow_credentials") is False
    assert settings.cors_origins, "an empty origin list answers no preflight at all"

    origin = settings.cors_origins[0]
    preflight = client.options(
        f"{API}/auth/login",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )
    assert preflight.status_code in (200, 204), preflight.text
    assert "access-control-allow-credentials" not in {
        key.lower() for key in preflight.headers
    }
    assert preflight.headers["access-control-allow-origin"] == origin


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_upload_ffprobe_gate_refuses_lying_magic(client, auth_headers):
    """The second gate: header bytes that say mp4 around a text body pass the
    magic table but ffprobe refuses the bytes — the upload loses."""
    lying = b"\x00\x00\x00\x20ftypisom" + b"just text pretending to be media " * 8
    resp = client.post(
        f"{API}/videos/upload",
        files={"file": ("liar.mp4", lying, "video/mp4")},
        headers=auth_headers,
    )
    assert resp.status_code == 415, resp.text
    body = resp.json()
    assert body["error"]["details"][0]["issue"] == "unsupported_container"
    with SessionLocal() as db:
        assert db.execute(select(Video)).scalars().all() == []


def test_magic_sniff_accepts_leading_boxes_short_heads_and_refuses_text():
    """The loosened probe: an 8-byte head already reaching the `ftyp` box and
    an `ftyp` beyond the first box both identify a container — while text
    bytes and impossible fragments stay mismatches. The gate does not weaken."""
    from app.routers.videos import _magic_mismatch, _sniff_family

    padded = b"\x00\x00\x00\x10free" + b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8
    assert _sniff_family(padded) == "isombx"
    assert _magic_mismatch(".mp4", padded) is False
    assert _magic_mismatch(".mp4", b"\x00\x00\x00\x20ftyp") is False  # 8 bytes
    assert _magic_mismatch(".mp4", b"<html>") is True
    assert _magic_mismatch(".mp4", b"\x00\x00\x00") is True  # too small to probe


# ------------------------------------------------------------------ P1 fixes


def test_passcode_throttle_key_normalises_the_folder_name(client, monkeypatch):
    """The lookup matches names case-insensitively and trimmed; the throttle
    key must be the same string, or re-spelling the target mints a new bucket
    while every spelling hits the same rows."""
    make_folder(client, operator(), name="Case Test")
    rate_limiter.reset()
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "public_redeem_rate_limit", 2)
    monkeypatch.setattr(app_settings, "public_redeem_rate_window", 60)
    try:
        responses = [
            client.post(
                f"{API}/public/passcodes/redeem",
                json={"code": "AAAA-BBBB", "folder_name": name},
            )
            for name in ("Case Test", "case test ", "CASE TEST")
        ]
        assert [r.status_code for r in responses[:2]] == [401, 401]
        # Same normalised target: the third spelling lands in the first
        # spelling's bucket instead of opening a fresh one.
        assert responses[2].status_code == 429
        assert responses[2].json()["error"]["code"] == "rate_limited"
    finally:
        rate_limiter.reset()


def test_cgnat_source_url_is_refused():
    """100.64.0.0/10 is none of private/loopback/link-local/reserved, so it
    used to pass; it must now be refused as a non-global address."""
    with pytest.raises(AppError) as excinfo:
        validate_source_url("http://100.64.0.1/x")
    assert excinfo.value.status_code == 422
    assert excinfo.value.details[0]["issue"] == "blocked_host"
    # A normal public address still validates: `is_global` is what allows it.
    assert validate_source_url("http://8.8.8.8/x") == "http://8.8.8.8/x"


def test_redeem_scan_limit_is_configurable(client, monkeypatch):
    """The candidate cap reads `public_redeem_scan_limit`: with it at 2 a
    no-target wrong guess bcrypt-verifies at most two live rows."""
    folder = make_folder(client, operator())
    actor = f"super_admin:{ADMIN_EMAIL}"
    with SessionLocal() as db:
        base = datetime.now(timezone.utc) - timedelta(minutes=10)
        for i in range(3):  # three live rows, newest-first window of 2
            db.add(
                FolderPasscode(
                    folder_id=folder["id"],
                    code_hash=hashlib.sha256(f"LIVE-{i:04d}".encode()).hexdigest(),
                    created_by=actor,
                    created_at=base + timedelta(seconds=i),
                )
            )
        db.commit()

    import app.routers.passcodes as passcodes_module

    calls: list[str] = []
    real_verify = passcodes_module.verify_code

    def _counting_verify(code, stored_hash):
        calls.append(stored_hash)
        return real_verify(code, stored_hash)

    monkeypatch.setattr(passcodes_module, "verify_code", _counting_verify)
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "public_redeem_scan_limit", 2)
    rate_limiter.reset()
    try:
        resp = client.post(f"{API}/public/passcodes/redeem", json={"code": "WRONG-9999"})
        assert resp.status_code == 401
        assert len(calls) == 2  # capped, not one per live row
    finally:
        rate_limiter.reset()


def test_merge_error_is_sane_at_the_boundary(client, auth_headers):
    """`MergeOut.error` was the last raw error field: a worker exception with
    a path and multi-line stderr comes back as one bounded, path-free line."""
    video_id = client.post(
        f"{API}/videos", json={"source_url": "https://example.org/merge-boom"}, headers=auth_headers
    ).json()["id"]
    stored = (
        "merge failed: /Users/mac/server/storage/media/"
        f"{video_id}/merged.wav: Invalid data found\n"
        "[mov,mp4 @ 0x7fa31c004380] stream 0: offset changed\n"
        "Error splitting the input into NAL units."
    )
    with SessionLocal() as db:
        db.add(Merge(video_id=video_id, status="failed", error=stored))
        db.commit()
        # The stored value is untouched — only the response is sanitised.
        row = db.execute(select(Merge).where(Merge.video_id == video_id)).scalar_one()
        assert row.error == stored

    listing = client.get(f"{API}/videos/{video_id}/merges", headers=auth_headers).json()
    error = listing["items"][0]["error"]
    assert "\n" not in error
    assert "/Users/" not in error
    assert not error.startswith("/")
    assert "[path]" in error
    assert len(error) <= 300