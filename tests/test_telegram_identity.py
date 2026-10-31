"""A Telegram person is not the bot: `telegram:<uid>` is not `telegram:bot`.

Regression for the privilege escalation where every principal with role
`telegram` was granted editor everywhere. Both the bot's service identity and
a person acting through it carried that role, so any Telegram user who knew a
video UUID could read and download another user's media and audio, read and
overwrite their transcript, and list the whole library.

The fix splits the two by actor: only `telegram:bot` keeps the blanket grant;
`telegram:<uid>` resolves to the linked active console account's role, or to a
bare `telegram` principal with no blanket access. These tests pin the contrast
from both sides — the person is refused, the bot still works — plus the
server-side link resolution and the actor-based consent gate.
"""
from __future__ import annotations

from uuid import uuid4

from app.db import SessionLocal
from app.models import AdminUser
from app.services import storage
from tests.helpers import (
    bot_headers,
    make_folder,
    make_transcript,
    telegram_user_headers,
    upload_video,
)

API = "/api/v1"

# An id linked to nothing: the plain "a Telegram user pasted a link" case.
UNLINKED_UID = 555001


def _operator_private_video(client, admin):
    """A unit in the operator's private folder — invisible to everyone else."""
    folder = make_folder(client, admin, name="Vault", visibility="private")
    return folder, upload_video(
        client, admin, folder_id=folder["id"], title="Vault clip"
    )


def _link(uid: int, role: str) -> str:
    """Bind `uid` to an active console account of `role`, as /link does."""
    email = f"tg-{role}-{uuid4().hex[:8]}@test.local"
    with SessionLocal() as db:
        db.add(
            AdminUser(
                email=email,
                role=role,
                is_active=True,
                telegram_user_id=uid,
            )
        )
        db.commit()
    return email


# ------------------------------------------------- media / audio / transcript


def test_a_telegram_person_cannot_read_another_users_unit(client, auth_headers):
    """Media, audio, transcript read and transcript edit all stay closed to a
    plain telegram:<uid>; the bot service identity keeps every one of them."""
    video = upload_video(client, auth_headers, title="Someone else's clip")
    make_transcript(client, auth_headers, video["id"], "private words")
    storage.audio_file(video["id"]).write_bytes(b"RIFF" + b"\x00" * 32)

    person = telegram_user_headers(UNLINKED_UID)

    assert (
        client.get(f"{API}/videos/{video['id']}/media", headers=person).status_code
        == 404
    )
    assert (
        client.get(f"{API}/videos/{video['id']}/audio", headers=person).status_code
        == 404
    )
    assert (
        client.get(f"{API}/videos/{video['id']}/transcript", headers=person).status_code
        == 403
    )
    edit = client.put(
        f"{API}/videos/{video['id']}/transcript",
        json={"content": "stolen"},
        headers=person,
    )
    assert edit.status_code == 403, edit.text

    # The very same routes still answer the bot's service identity.
    bot = bot_headers()
    assert client.get(f"{API}/videos/{video['id']}/media", headers=bot).status_code == 200
    assert client.get(f"{API}/videos/{video['id']}/audio", headers=bot).status_code == 200
    assert (
        client.get(f"{API}/videos/{video['id']}/transcript", headers=bot).status_code
        == 200
    )
    edit = client.put(
        f"{API}/videos/{video['id']}/transcript",
        json={"content": "corrected by the bot"},
        headers=bot,
    )
    assert edit.status_code == 201, edit.text


# --------------------------------------------------------------- the listings


def test_a_telegram_person_does_not_get_the_library_listing(client, auth_headers):
    """`GET /dataset/cards` (bare) and `GET /videos` are empty for a plain
    telegram:<uid>: the operator's private unit is not in either, and the bot
    still lists everything."""
    _, video = _operator_private_video(client, auth_headers)
    person = telegram_user_headers(UNLINKED_UID)

    cards = client.get(f"{API}/dataset/cards", headers=person)
    assert cards.status_code == 200, cards.text
    assert cards.json()["items"] == [] and cards.json()["total"] == 0, cards.text

    listing = client.get(
        f"{API}/videos", params={"page_size": 1000}, headers=person
    )
    assert listing.status_code == 200, listing.text
    body = listing.json()
    assert body["items"] == [] and body["total"] == 0, body

    bot = bot_headers()
    bot_cards = client.get(
        f"{API}/dataset/cards", params={"page_size": 100}, headers=bot
    ).json()
    assert video["id"] in {c["video_id"] for c in bot_cards["items"]}, bot_cards
    bot_listing = client.get(
        f"{API}/videos", params={"page_size": 1000}, headers=bot
    ).json()
    assert video["id"] in {v["id"] for v in bot_listing["items"]}, bot_listing


# --------------------------------------------------------- linked console role


def test_a_linked_admin_resolves_server_side_to_admin(client, auth_headers):
    """A uid linked to an active admin acts as that admin — while its actor,
    the ownership key, stays exactly `telegram:<uid>`."""
    _, video = _operator_private_video(client, auth_headers)
    uid = 555101
    admin_email = _link(uid, "admin")
    person = telegram_user_headers(uid)

    me = client.get(f"{API}/auth/me", headers=person).json()
    assert me["actor"] == f"telegram:{uid}", me
    assert me["role"] == "admin" and me["email"] == admin_email, me

    # Editor everywhere: reads the private unit and edits its transcript.
    assert (
        client.get(f"{API}/videos/{video['id']}", headers=person).status_code == 200
    )
    edit = client.put(
        f"{API}/videos/{video['id']}/transcript",
        json={"content": "admin correction"},
        headers=person,
    )
    assert edit.status_code == 201, edit.text


def test_a_linked_ordinary_user_does_not_become_an_editor(client, auth_headers):
    """A link grants the account's role, not a promotion: role `user` stays
    scoped to their own ingests and public folders."""
    _, video = _operator_private_video(client, auth_headers)
    uid = 555102
    _link(uid, "user")
    person = telegram_user_headers(uid)

    me = client.get(f"{API}/auth/me", headers=person).json()
    assert me["actor"] == f"telegram:{uid}" and me["role"] == "user", me
    assert (
        client.get(f"{API}/videos/{video['id']}", headers=person).status_code == 403
    )


# ------------------------------------------------------------- consent gate


def test_a_linked_admin_still_hits_the_consent_gate(client):
    """The TikTok/Facebook consent gate keys on the ACTOR, not the role, so a
    linked admin cannot skip the cookies requirement by resolving as admin."""
    uid = 555201
    _link(uid, "admin")
    resp = client.post(
        f"{API}/videos",
        json={
            "source_url": "https://www.tiktok.com/@gate/video/7000000000000000201"
        },
        headers=telegram_user_headers(uid),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "consent_required", resp.text
