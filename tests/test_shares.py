"""Share links: a revocable URL that grants a role without an account.

The security-relevant properties are that a link is *scoped* (it opens one
folder, not the library), *revocable* (revoking stops it working immediately)
and *expiring* (a dead link reveals nothing about whether it ever existed).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.db import SessionLocal
from app.models import ShareLink
from app.routers.shares import find_active_share, folder_for_share, share_url
from tests.helpers import ensure_member, make_folder, member_headers


def _share(client, headers, folder_id, **payload):
    return client.post(
        f"/api/v1/folders/{folder_id}/shares", json=payload or {}, headers=headers
    )


def _make_share(client, headers, folder_id, **payload):
    resp = _share(client, headers, folder_id, **payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ------------------------------------------------------------------ creating


def test_create_share_returns_a_token_and_a_portal_url(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    body = _make_share(client, auth_headers, folder["id"])

    assert body["token"]
    assert body["role"] == "viewer"
    assert body["folder_id"] == folder["id"]
    assert body["revoked_at"] is None
    # The link points at the guest portal, not the operator console.
    assert body["url"].startswith("http://localhost:5174/share/")
    assert body["token"] in body["url"]


def test_share_url_follows_the_configured_portal(client, auth_headers, monkeypatch):
    from app.config import settings
    from app.routers import shares

    monkeypatch.setattr(settings, "portal_frontend_url", "https://data.example.org/")
    assert share_url("abc") == "https://data.example.org/share/abc"
    assert shares.share_url("abc") == "https://data.example.org/share/abc"


def test_a_share_defaults_to_read_only(client, auth_headers):
    """The low-friction option must not be able to edit by accident."""
    folder = make_folder(client, auth_headers, name="Dataset")
    body = _make_share(client, auth_headers, folder["id"])
    assert body["role"] == "viewer"


def test_an_editor_share_is_possible(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    body = _make_share(client, auth_headers, folder["id"], role="editor")
    assert body["role"] == "editor"


def test_share_role_is_validated(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    resp = _share(client, auth_headers, folder["id"], role="owner")
    assert resp.status_code == 422


def test_two_shares_get_distinct_tokens(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    a = _make_share(client, auth_headers, folder["id"])
    b = _make_share(client, auth_headers, folder["id"])
    assert a["token"] != b["token"]


def test_creating_a_share_requires_authentication(client):
    resp = _share(client, {}, "00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 401


def test_creating_a_share_requires_manage_on_the_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    resp = _share(client, member_headers(email), folder["id"])
    assert resp.status_code == 403


def test_creating_a_share_on_an_unknown_folder_is_a_404(client, auth_headers):
    from uuid import uuid4

    resp = _share(client, auth_headers, uuid4())
    assert resp.status_code == 404


# ----------------------------------------------------------------- listing


def test_list_shares_returns_them_newest_first(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    first = _make_share(client, auth_headers, folder["id"])
    second = _make_share(client, auth_headers, folder["id"])

    resp = client.get(f"/api/v1/folders/{folder['id']}/shares", headers=auth_headers)
    assert resp.status_code == 200
    tokens = [s["token"] for s in resp.json()]
    assert tokens[0] == second["token"]
    assert tokens[1] == first["token"]


def test_listing_shares_requires_manage(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    resp = client.get(
        f"/api/v1/folders/{folder['id']}/shares", headers=member_headers(email)
    )
    assert resp.status_code == 403


def test_describe_share_labels_a_link_before_it_is_sent(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    share = _make_share(client, auth_headers, folder["id"])
    resp = client.get(f"/api/v1/shares/{share['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["token"] == share["token"]


def test_describe_unknown_share_is_a_404(client, auth_headers):
    from uuid import uuid4

    resp = client.get(f"/api/v1/shares/{uuid4()}", headers=auth_headers)
    assert resp.status_code == 404


# ----------------------------------------------------------------- revoking


def test_revoking_a_share_stops_it_working(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    share = _make_share(client, auth_headers, folder["id"])

    revoked = client.delete(f"/api/v1/shares/{share['id']}", headers=auth_headers)
    assert revoked.status_code == 204

    with SessionLocal() as db:
        assert find_active_share(db, share["token"]) is None


def test_a_revoked_share_cannot_open_its_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    share = _make_share(client, auth_headers, folder["id"])
    client.delete(f"/api/v1/shares/{share['id']}", headers=auth_headers)

    with SessionLocal() as db:
        from app.errors import AppError

        with pytest.raises(AppError) as denied:
            folder_for_share(db, share["token"])
        assert denied.value.status_code == 404


def test_revoking_requires_manage_on_the_share_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    share = _make_share(client, auth_headers, folder["id"])
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    resp = client.delete(f"/api/v1/shares/{share['id']}", headers=member_headers(email))
    assert resp.status_code == 403


def test_revoking_twice_is_harmless(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    share = _make_share(client, auth_headers, folder["id"])
    assert client.delete(f"/api/v1/shares/{share['id']}", headers=auth_headers).status_code == 204
    assert client.delete(f"/api/v1/shares/{share['id']}", headers=auth_headers).status_code == 204


# ------------------------------------------------------------------ expiry


def test_an_expired_share_is_no_longer_active(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    share = _make_share(client, auth_headers, folder["id"], expires_at=past)

    with SessionLocal() as db:
        assert find_active_share(db, share["token"]) is None


def test_a_share_can_be_created_already_expired(client, auth_headers):
    """Creating an expired link is allowed; it simply never opens anything."""
    folder = make_folder(client, auth_headers, name="Dataset")
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    resp = _share(client, auth_headers, folder["id"], expires_at=past)
    assert resp.status_code == 201


def test_a_future_expiry_keeps_the_share_active(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    share = _make_share(client, auth_headers, folder["id"], expires_at=future)
    with SessionLocal() as db:
        assert find_active_share(db, share["token"]) is not None


# ----------------------------------------------------------------- resolving


def test_folder_for_share_returns_the_folder_for_a_valid_token(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    share = _make_share(client, auth_headers, folder["id"])
    with SessionLocal() as db:
        assert str(folder_for_share(db, share["token"]).id) == folder["id"]


def test_an_unknown_token_is_indistinguishable_from_a_dead_one(client, auth_headers):
    with SessionLocal() as db:
        from app.errors import AppError

        for token in ("never-existed", "", "a" * 43):
            with pytest.raises(AppError) as denied:
                folder_for_share(db, token)
            assert denied.value.status_code == 404
            assert denied.value.code == "not_found"


def test_a_share_resolves_only_to_its_own_folder(client, auth_headers):
    """The token names one folder; it is not a key to the library."""
    a = make_folder(client, auth_headers, name="A")
    b = make_folder(client, auth_headers, name="B")
    share_a = _make_share(client, auth_headers, a["id"])

    with SessionLocal() as db:
        assert str(folder_for_share(db, share_a["token"]).id) == a["id"]
        assert str(folder_for_share(db, share_a["token"]).id) != b["id"]


def test_revoking_a_share_does_not_affect_another_folder(client, auth_headers):
    a = make_folder(client, auth_headers, name="A")
    b = make_folder(client, auth_headers, name="B")
    share_a = _make_share(client, auth_headers, a["id"])
    share_b = _make_share(client, auth_headers, b["id"])

    client.delete(f"/api/v1/shares/{share_a['id']}", headers=auth_headers)

    with SessionLocal() as db:
        assert find_active_share(db, share_b["token"]) is not None
        assert db.get(ShareLink, share_b["id"]).revoked_at is None
