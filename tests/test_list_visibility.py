"""List visibility: a row the caller cannot open is not listed at all.

`GET /videos` and `GET /dataset/cards` used to include rows of private folders
the developer/reviewer tiers hold no role on, marked `can_open: false`. The
operator's decision is that such rows are hidden entirely, so the listing never
offers a dialog that would only 403. The tests pin the contract from both
sides: the filtered caller no longer sees the private unit, the admin tier
still does, and `total`/pagination count exactly the rows that survived the
filter.
"""
from __future__ import annotations

from uuid import uuid4

from app.auth import hash_password
from app.db import SessionLocal
from app.models import AdminUser
from tests.helpers import make_folder, upload_video

API = "/api/v1"


def _role_headers(client, role: str) -> dict:
    """A signed-in account with the given role.

    Roles are database-authoritative, so logging in normally is the honest way
    to mint the token: whatever the JWT claims, the row decides.
    """
    email = f"{role}-{uuid4().hex[:8]}@test.local"
    with SessionLocal() as db:
        db.add(
            AdminUser(
                email=email,
                role=role,
                password_hash=hash_password("role-pass-12345"),
                is_active=True,
            )
        )
        db.commit()
    resp = client.post(
        f"{API}/auth/login", json={"email": email, "password": "role-pass-12345"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _dataset(client, admin):
    """A private folder the seeded super admin keeps to themselves, a public
    one, and a video in each — the invisible row and its visible contrast."""
    private = make_folder(client, admin, name="Vault", visibility="private")
    hidden = upload_video(client, admin, folder_id=private["id"], title="Hidden clip")
    public = make_folder(client, admin, name="Open", visibility="public")
    shown = upload_video(client, admin, folder_id=public["id"], title="Public clip")
    return private, hidden, shown


def test_developer_video_list_hides_unopenable_rows(client, auth_headers):
    """The developer sees the public unit and their own ingest, never the
    private folder's — a row that would only 403 on open is not listed."""
    private, hidden, shown = _dataset(client, auth_headers)
    dev = _role_headers(client, "developer")
    own = upload_video(client, dev, title="Dev clip")

    body = client.get(f"{API}/videos", params={"page_size": 1000}, headers=dev).json()
    ids = {v["id"] for v in body["items"]}
    assert hidden["id"] not in ids, body
    assert shown["id"] in ids, body
    assert own["id"] in ids, body
    assert body["total"] == len(body["items"]) == 2, body
    assert all(v["can_open"] for v in body["items"]), body


def test_admin_video_list_still_shows_the_private_folder(client, auth_headers):
    """The filter is a no-op for the admin tier: the private unit stays listed
    and stays openable."""
    private, hidden, shown = _dataset(client, auth_headers)

    body = client.get(f"{API}/videos", params={"page_size": 1000}, headers=auth_headers).json()
    ids = {v["id"] for v in body["items"]}
    assert hidden["id"] in ids, body
    assert shown["id"] in ids, body
    assert body["total"] == len(body["items"]) == 2, body
    assert all(v["can_open"] for v in body["items"]), body


def test_developer_cards_hide_the_private_folder(client, auth_headers):
    """Same contract on the dataset cards: the private folder's card is gone
    for the developer, present for the admin — and the folder id itself
    stays a 404 for the developer, so the cards cannot be probed by scope."""
    private, hidden, shown = _dataset(client, auth_headers)
    dev = _role_headers(client, "developer")
    own = upload_video(client, dev, title="Dev clip")

    body = client.get(f"{API}/dataset/cards", params={"page_size": 100}, headers=dev).json()
    ids = {c["video_id"] for c in body["items"]}
    assert hidden["id"] not in ids, body
    assert shown["id"] in ids, body
    assert own["id"] in ids, body
    assert body["total"] == len(body["items"]) == 2, body
    assert all(c["can_open"] for c in body["items"]), body

    admin_cards = client.get(
        f"{API}/dataset/cards", params={"folder_id": private["id"]}, headers=auth_headers
    ).json()
    assert hidden["id"] in {c["video_id"] for c in admin_cards["items"]}, admin_cards

    probed = client.get(
        f"{API}/dataset/cards", params={"folder_id": private["id"]}, headers=dev
    )
    assert probed.status_code == 404, probed.text


def test_filtered_pages_stay_full_and_total_counts_openable_rows(client, auth_headers):
    """Pagination coherence after the filter: `total` counts only the rows the
    developer could open, pages never underfill because a hidden row no longer
    consumes an offset, and a page size past the end of the data yields
    `len(items) == total`."""
    private = make_folder(client, auth_headers, name="Vault", visibility="private")
    hidden_ids = {
        upload_video(client, auth_headers, folder_id=private["id"], title=f"Hidden {i}")["id"]
        for i in range(3)
    }
    public = make_folder(client, auth_headers, name="Open", visibility="public")
    for i in range(2):
        upload_video(client, auth_headers, folder_id=public["id"], title=f"Public {i}")
    dev = _role_headers(client, "developer")
    own = upload_video(client, dev, title="Dev clip")

    # Three openable rows for the developer: two public units and their own.
    first = client.get(f"{API}/videos", params={"page": 1, "page_size": 2}, headers=dev).json()
    second = client.get(f"{API}/videos", params={"page": 2, "page_size": 2}, headers=dev).json()
    assert first["total"] == second["total"] == 3, (first, second)
    assert len(first["items"]) == 2, first
    assert len(second["items"]) == 1, second
    assert not hidden_ids & {v["id"] for v in first["items"] + second["items"]}

    whole = client.get(f"{API}/videos", params={"page_size": 1000}, headers=dev).json()
    assert len(whole["items"]) == whole["total"] == 3, whole

    cards = client.get(f"{API}/dataset/cards", params={"page_size": 100}, headers=dev).json()
    assert len(cards["items"]) == cards["total"] == 3, cards
    assert not hidden_ids & {c["video_id"] for c in cards["items"]}