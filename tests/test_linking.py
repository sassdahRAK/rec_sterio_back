"""Telegram <-> console account linking.

Covers the console half (POST/GET/DELETE /auth/telegram-link), the bot half
(POST /bot-users/{id}/link, GET /bot-users/admin-ids), the one-to-one pairing
conflicts in both directions, and the display-name resolution a video detail
view performs for a linked Telegram actor.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import AdminUser, LinkCode
from app.routers.bot_users import CONSENT_VERSION
from app.routers.passcodes import CODE_ALPHABET, hash_code
from tests.conftest import ADMIN_EMAIL
from tests.helpers import (
    ensure_user,
    guest_headers,
    user_headers,
    telegram_user_headers,
)

UID = 77001
OTHER_UID = 77002
USER_UID = 77003
INACTIVE_ADMIN_UID = 77004


# ------------------------------------------------------------------ helpers


def _create_code(client, headers=None) -> dict:
    resp = client.post("/api/v1/auth/telegram-link", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _redeem(client, uid: int, code: str, confirm: bool):
    return client.post(
        f"/api/v1/bot-users/{uid}/link",
        json={"code": code, "confirm": confirm},
        headers=telegram_user_headers(uid),
    )


def _admin_row(email: str = ADMIN_EMAIL) -> AdminUser | None:
    with SessionLocal() as db:
        return db.execute(
            select(AdminUser).where(AdminUser.email == email)
        ).scalar_one_or_none()


def _link(client, headers, uid: int) -> None:
    """The full round trip: a fresh code, confirmed by `uid`."""
    code = _create_code(client, headers)["code"]
    resp = _redeem(client, uid, code, confirm=True)
    assert resp.status_code == 200, resp.text


def _admin_ids(client) -> dict:
    resp = client.get(
        "/api/v1/bot-users/admin-ids",
        headers={"Authorization": f"Bearer {settings.bot_api_token}"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------ console side


def test_create_link_code_requires_a_console_account(client, auth_headers):
    # No token at all.
    resp = client.post("/api/v1/auth/telegram-link")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
    # A guest browses public datasets; it owns no account to link.
    resp = client.post("/api/v1/auth/telegram-link", headers=guest_headers())
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"
    # The bot's service identity (token alone) is not a console account either.
    resp = client.post(
        "/api/v1/auth/telegram-link",
        headers={"Authorization": f"Bearer {settings.bot_api_token}"},
    )
    assert resp.status_code == 403
    # Nor is a forwarded Telegram user.
    resp = client.post(
        "/api/v1/auth/telegram-link", headers=telegram_user_headers(UID)
    )
    assert resp.status_code == 403
    # A signed-in admin may.
    assert _create_code(client, auth_headers)["code"]


def test_create_link_code_shape_and_ttl(client, auth_headers):
    body = _create_code(client, auth_headers)
    assert len(body["code"]) == 8
    assert all(ch in CODE_ALPHABET for ch in body["code"])
    # No bot username is configured in the test environment.
    assert body["deep_link"] is None
    expires = datetime.fromisoformat(body["expires_at"])
    remaining = expires - datetime.now(timezone.utc)
    assert timedelta(minutes=13) <= remaining <= timedelta(minutes=16)
    # Unlinked so far.
    resp = client.get("/api/v1/auth/telegram-link", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == {"telegram_user_id": None}


def test_deep_link_is_built_when_a_bot_username_is_configured(
    client, auth_headers, monkeypatch
):
    monkeypatch.setattr(settings, "telegram_bot_username", "recsterio_test_bot")
    body = _create_code(client, auth_headers)
    assert body["deep_link"] == (
        f"https://t.me/recsterio_test_bot?start=link-{body['code']}"
    )


def test_creating_a_code_revokes_the_previous_outstanding_one(
    client, auth_headers
):
    first = _create_code(client, auth_headers)["code"]
    second = _create_code(client, auth_headers)["code"]
    assert first != second
    # The older code was retired at creation: only one live code per account.
    resp = _redeem(client, UID, first, confirm=False)
    assert resp.status_code == 401
    resp = _redeem(client, UID, second, confirm=False)
    assert resp.status_code == 200
    assert resp.json() == {"account_email": ADMIN_EMAIL}


def test_unlink_status_and_delete_require_a_console_account(client):
    for method in ("get", "delete"):
        resp = getattr(client, method)("/api/v1/auth/telegram-link")
        assert resp.status_code == 401
        resp = getattr(client, method)(
            "/api/v1/auth/telegram-link", headers=guest_headers()
        )
        assert resp.status_code == 403


# ----------------------------------------------------------------- bot side


def test_dry_run_names_the_account_without_linking_or_consuming(
    client, auth_headers
):
    code = _create_code(client, auth_headers)["code"]
    resp = _redeem(client, UID, code, confirm=False)
    assert resp.status_code == 200
    assert resp.json() == {"account_email": ADMIN_EMAIL}
    # Not linked...
    assert _admin_row().telegram_user_id is None
    # ...and not consumed: the dry run can be repeated.
    resp = _redeem(client, UID, code, confirm=False)
    assert resp.status_code == 200


def test_confirm_links_the_account_and_redeems_the_code_once(
    client, auth_headers
):
    code = _create_code(client, auth_headers)["code"]
    resp = _redeem(client, UID, code, confirm=True)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["account_email"] == ADMIN_EMAIL
    assert body["account_name"] is None  # the seeded admin has no name
    assert body["account_role"] == "super_admin"
    assert _admin_row().telegram_user_id == UID

    # One-time: the redeem revoked the code, and every later attempt — dry run
    # or confirm — is the same indistinguishable 401 as for any dead code.
    for confirm in (False, True):
        resp = _redeem(client, UID, code, confirm=confirm)
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "unauthorized"
        assert resp.json()["error"]["message"] == "That link code is not valid."


def test_wrong_code_is_indistinguishable_from_no_code(client, auth_headers):
    _create_code(client, auth_headers)  # a live code exists; this is not it
    for confirm in (False, True):
        resp = _redeem(client, UID, "AB23CD45", confirm=confirm)
        assert resp.status_code == 401
        assert resp.json()["error"]["message"] == "That link code is not valid."


def test_expired_code_is_not_valid(client, auth_headers):
    admin = _admin_row()
    with SessionLocal() as db:
        db.add(
            LinkCode(
                code_hash=hash_code("AB23CD45"),
                admin_user_id=admin.id,
                expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            )
        )
        db.commit()
    resp = _redeem(client, UID, "AB23CD45", confirm=False)
    assert resp.status_code == 401
    assert resp.json()["error"]["message"] == "That link code is not valid."


def test_redeem_requires_the_bot_credentials(client, auth_headers):
    code = _create_code(client, auth_headers)["code"]
    # No service token.
    resp = client.post(
        f"/api/v1/bot-users/{UID}/link",
        json={"code": code, "confirm": False},
        headers={"X-Requested-By": f"telegram:{UID}"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "not_authenticated"
    # Right token, but acting as somebody else.
    resp = client.post(
        f"/api/v1/bot-users/{UID}/link",
        json={"code": code, "confirm": False},
        headers=telegram_user_headers(OTHER_UID),
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


def test_conflict_when_the_telegram_id_is_claimed_by_another_account(
    client, auth_headers
):
    _link(client, auth_headers, UID)
    email = ensure_user("linktarget")
    code = _create_code(client, user_headers(email))["code"]
    # UID tries to link to the user while the admin already holds it.
    resp = _redeem(client, UID, code, confirm=True)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "already_linked"
    # No email is disclosed.
    assert email not in resp.text


def test_conflict_when_the_account_is_linked_to_a_different_telegram_user(
    client, auth_headers
):
    _link(client, auth_headers, UID)
    code = _create_code(client, auth_headers)["code"]
    # OTHER_UID tries to claim an account that UID already holds.
    resp = _redeem(client, OTHER_UID, code, confirm=True)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "already_linked"
    # The pairing is unchanged.
    assert _admin_row().telegram_user_id == UID


def test_self_relink_overwrites_the_same_pairing(client, auth_headers):
    _link(client, auth_headers, UID)
    code = _create_code(client, auth_headers)["code"]
    resp = _redeem(client, UID, code, confirm=True)
    assert resp.status_code == 200
    assert _admin_row().telegram_user_id == UID


def test_admin_ids_list_active_admin_links_only(client, auth_headers):
    # The service token alone is enough — no X-Requested-By applies.
    assert _admin_ids(client) == {"ids": []}
    _link(client, auth_headers, UID)
    assert _admin_ids(client) == {"ids": [UID]}

    with SessionLocal() as db:
        db.add(
            AdminUser(
                email="linked-user@test.local",
                role="user",
                telegram_user_id=USER_UID,
            )
        )
        db.add(
            AdminUser(
                email="sleeping-admin@test.local",
                role="admin",
                is_active=False,
                telegram_user_id=INACTIVE_ADMIN_UID,
            )
        )
        db.commit()
    # A user link and an inactive admin link are both invisible.
    assert _admin_ids(client) == {"ids": [UID]}

    resp = client.get("/api/v1/bot-users/admin-ids")
    assert resp.status_code == 401


def test_unlink_clears_the_link(client, auth_headers):
    _link(client, auth_headers, UID)
    resp = client.get("/api/v1/auth/telegram-link", headers=auth_headers)
    assert resp.json() == {"telegram_user_id": UID}

    resp = client.delete("/api/v1/auth/telegram-link", headers=auth_headers)
    assert resp.status_code == 204
    resp = client.get("/api/v1/auth/telegram-link", headers=auth_headers)
    assert resp.json() == {"telegram_user_id": None}
    assert UID not in _admin_ids(client)["ids"]
    # Idempotent: unlinking again is a no-op.
    resp = client.delete("/api/v1/auth/telegram-link", headers=auth_headers)
    assert resp.status_code == 204


def test_bot_user_response_carries_the_linked_account(client, auth_headers):
    resp = client.post(
        f"/api/v1/bot-users/{UID}/consent",
        json={"consent_version": CONSENT_VERSION},
        headers=telegram_user_headers(UID),
    )
    assert resp.status_code == 200, resp.text

    got = client.get(
        f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID)
    )
    assert got.status_code == 200
    body = got.json()
    assert body["linked_email"] is None
    assert body["linked_name"] is None
    assert body["linked_role"] is None

    _link(client, auth_headers, UID)
    got = client.get(
        f"/api/v1/bot-users/{UID}", headers=telegram_user_headers(UID)
    )
    body = got.json()
    assert body["linked_email"] == ADMIN_EMAIL
    assert body["linked_name"] is None
    assert body["linked_role"] == "super_admin"


# ----------------------------------------------------- display resolution


def test_video_detail_resolves_linked_actor_display_names(
    client, auth_headers
):
    _link(client, auth_headers, UID)

    # Ingested by the linked Telegram user: the actor string is raw, the
    # display is resolved from the account link.
    url = f"https://www.youtube.com/watch?v=link-{uuid4().hex[:10]}"
    resp = client.post(
        "/api/v1/videos", json={"source_url": url}, headers=telegram_user_headers(UID)
    )
    assert resp.status_code == 201, resp.text
    video = resp.json()
    assert video["created_by"] == f"telegram:{UID}"

    detail = client.get(f"/api/v1/videos/{video['id']}", headers=auth_headers)
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["created_by"] == f"telegram:{UID}"  # never rewritten
    assert body["created_by_display"] == ADMIN_EMAIL
    assert body["collected_by_display"] is None

    # A TvNa record credits the collector by the same actor form.
    resp = client.post(
        f"/api/v1/videos/{video['id']}/tvna",
        json={"location_tag": "Phnom Penh", "collected_by": f"telegram:{UID}"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    detail = client.get(f"/api/v1/videos/{video['id']}", headers=auth_headers)
    body = detail.json()
    assert body["collected_by"] == f"telegram:{UID}"
    assert body["collected_by_display"] == ADMIN_EMAIL

    # An unlinked actor form resolves to nothing — the raw string stays.
    url = f"https://www.youtube.com/watch?v=link-{uuid4().hex[:10]}"
    resp = client.post(
        "/api/v1/videos", json={"source_url": url}, headers=auth_headers
    )
    assert resp.status_code == 201, resp.text
    detail = client.get(f"/api/v1/videos/{resp.json()['id']}", headers=auth_headers)
    body = detail.json()
    assert body["created_by"] == f"super_admin:{ADMIN_EMAIL}"
    assert body["created_by_display"] is None