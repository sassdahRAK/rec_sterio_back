"""Portal link: one signed token for the one video the bot just ingested.

A URL pasted to the bot is ingested *unfiled* and owned by `telegram:<uid>`, so
the old portal page resolved it to an empty card list (`app/access.py` returns
no role for an unfiled video). `POST /videos/{id}/portal-link` mints an ordinary
guest JWT that names exactly that id. These tests pin the whole boundary: the
token opens its own unit (including live progress), cannot reach any other id,
cannot list the library, and a tampered or expired copy fails the normal `401`.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import func, select

from app.auth import create_access_token, decode_token, issue_guest_token
from app.db import SessionLocal
from app.models import AdminUser, Video
from tests.helpers import (
    make_folder,
    scoped_headers,
    telegram_user_headers,
    upload_video,
)

API = "/api/v1"
# The uid every `_make_video` row is ingested by, unless a test says otherwise.
USER_UID = 123456


def _make_video(
    *,
    folder_id=None,
    status: str = "downloading",
    progress: int = 37,
    error: str = "download paused by the platform",
    created_by: str = f"telegram:{USER_UID}",
) -> str:
    """A bot-ingested row, unfiled by default — the shape the bot creates."""
    with SessionLocal() as db:
        video = Video(
            source_url=f"https://example.com/{uuid4().hex}",
            platform="youtube",
            title="Pasted link",
            status=status,
            progress=progress,
            error=error,
            folder_id=folder_id,
            created_by=created_by,
        )
        db.add(video)
        db.commit()
        return str(video.id)


def _mint(client, video_id: str, headers) -> dict:
    resp = client.post(f"{API}/videos/{video_id}/portal-link", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _video_count() -> int:
    with SessionLocal() as db:
        return db.execute(select(func.count()).select_from(Video)).scalar_one()


def test_portal_link_response_shape_and_claims(client):
    """The response names the one video, and the JWT is a normal guest token."""
    video_id = _make_video()

    before = _video_count()
    body = _mint(client, video_id, telegram_user_headers(USER_UID))
    # Signing is not persistence: the row count is unchanged.
    assert _video_count() == before

    assert set(body) == {"token", "expires_at", "path"}
    assert body["path"] == f"/video/{video_id}?t={body['token']}"
    assert body["expires_at"].endswith("Z")

    payload = decode_token(body["token"])
    assert payload["role"] == "guest"
    assert payload["sub"].startswith("guest:")
    assert payload["videos"] == [video_id]
    assert payload["video_roles"] == {video_id: "viewer"}
    assert payload["folders"] == []


def test_scoped_token_reads_its_unfiled_video_with_live_progress(client):
    """The whole point: an unfiled unit now resolves through `video_id`, with
    the same progress/error the worker is writing."""
    video_id = _make_video(progress=42, error="retry scheduled")
    token = _mint(client, video_id, telegram_user_headers(USER_UID))["token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.get(
        f"{API}/dataset/cards", params={"video_id": video_id}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1, body
    card = body["items"][0]
    assert card["video_id"] == video_id
    assert card["status"] == "downloading"
    assert card["progress"] == 42
    assert card["error"] == "retry scheduled"
    assert card["can_open"] is True

    # The detail route runs the same `video_role` check, so it agrees.
    detail = client.get(f"{API}/videos/{video_id}", headers=headers)
    assert detail.status_code == 200, detail.text


def test_scoped_token_cannot_reach_a_different_video(client):
    """The grant is exactly one id: another row is an empty card list and a
    403 on the detail route, even when both were ingested the same way."""
    mine = _make_video()
    other = _make_video()
    token = _mint(client, mine, telegram_user_headers(USER_UID))["token"]
    headers = {"Authorization": f"Bearer {token}"}

    cards = client.get(
        f"{API}/dataset/cards", params={"video_id": other}, headers=headers
    )
    assert cards.status_code == 200, cards.text
    assert cards.json() == {
        "items": [],
        "total": 0,
        "page": 1,
        "page_size": 20,
    }

    detail = client.get(f"{API}/videos/{other}", headers=headers)
    assert detail.status_code == 403, detail.text


def test_scoped_token_cannot_list_the_library(client, auth_headers):
    """No `video_id` means no listing at all — not even the scoped unit.

    The token can open its own video by id, but it must not enumerate, which
    is why the SQL prefilter admits the scoped id only when the request names
    it. A public folder (and the public unit inside it) must not widen the
    token either: `visible_folders` withholds public visibility from a
    video-scoped guest.
    """
    mine = _make_video()
    _make_video()
    public = make_folder(client, auth_headers, name="Public", visibility="public")
    _make_video(folder_id=public["id"])
    token = _mint(client, mine, telegram_user_headers(USER_UID))["token"]
    headers = {"Authorization": f"Bearer {token}"}

    body = client.get(
        f"{API}/dataset/cards", params={"page_size": 100}, headers=headers
    ).json()
    assert body == {"items": [], "total": 0, "page": 1, "page_size": 100}

    # Search shares that query, so it cannot surface the public unit either.
    searched = client.get(
        f"{API}/dataset/cards", params={"q": "Pasted"}, headers=headers
    ).json()
    assert searched["total"] == 0, searched

    # The public tree embeds each folder's cards, so it must not list either.
    tree = client.get(f"{API}/public/folders", headers=headers).json()
    assert tree == {"items": [], "total": 0}

    # The unit is still reachable when the request names it explicitly.
    own = client.get(
        f"{API}/dataset/cards", params={"video_id": mine}, headers=headers
    ).json()
    assert own["total"] == 1, own


def test_scoped_token_refused_by_other_listings_and_transcripts(
    client, auth_headers
):
    """The credential opens one video: the console list, the operator queue and
    every other unit's transcript stay closed, including a public-folder unit
    the token never named."""
    mine = _make_video()
    other = _make_video()
    public = make_folder(client, auth_headers, name="Public", visibility="public")
    public_video = _make_video(folder_id=public["id"])
    token = _mint(client, mine, telegram_user_headers(USER_UID))["token"]
    headers = {"Authorization": f"Bearer {token}"}

    videos = client.get(f"{API}/videos", params={"page_size": 100}, headers=headers)
    assert videos.status_code == 200, videos.text
    assert videos.json()["total"] == 0, videos.json()

    jobs = client.get(f"{API}/jobs", headers=headers)
    assert jobs.status_code == 403, jobs.text

    for other_id in (other, public_video):
        transcript = client.get(f"{API}/videos/{other_id}/transcript", headers=headers)
        assert transcript.status_code == 403, transcript.text


def test_scoped_token_reads_own_card_detail_and_media(client, bot_headers, auth_headers):
    """A filed unit is the regression case: the folder rule alone 404'd its
    media, but `_authorize_media` now asks `video_role`, which knows the scope."""
    folder = make_folder(client, auth_headers, name="Private", visibility="private")
    uploaded = upload_video(client, bot_headers, folder_id=folder["id"])
    video_id = uploaded["id"]
    # The upload ran as the bare service identity, so no Telegram user owns
    # it; the operator mints exactly as the console would.
    token = _mint(client, video_id, auth_headers)["token"]
    headers = {"Authorization": f"Bearer {token}"}

    card = client.get(
        f"{API}/dataset/cards", params={"video_id": video_id}, headers=headers
    )
    assert card.status_code == 200, card.text
    assert card.json()["total"] == 1, card.json()
    assert card.json()["items"][0]["video_id"] == video_id

    detail = client.get(f"{API}/videos/{video_id}", headers=headers)
    assert detail.status_code == 200, detail.text

    media = client.get(f"{API}/videos/{video_id}/media", headers=headers)
    assert media.status_code == 200, media.text
    assert media.content


def test_folder_scoped_token_still_lists_public_folders(client, auth_headers):
    """Regression guard for the public-visibility suppression: a passcode/share
    token has an empty `scoped_video_ids`, so public folders keep contributing
    exactly as they did."""
    public_a = make_folder(client, auth_headers, name="Public A", visibility="public")
    public_b = make_folder(client, auth_headers, name="Public B", visibility="public")
    _make_video(folder_id=public_a["id"])
    _make_video(folder_id=public_b["id"])

    headers = scoped_headers(public_a["id"])
    body = client.get(
        f"{API}/dataset/cards", params={"page_size": 100}, headers=headers
    ).json()
    # Scoped to A, but B is public: both public units stay visible.
    assert body["total"] == 2, body

    folders = client.get(f"{API}/public/folders", headers=headers).json()
    assert folders["total"] == 2, folders


def test_tampered_signature_is_rejected(client):
    """A hand-edited signature fails `decode_token` and yields the usual 401."""
    video_id = _make_video()
    token = _mint(client, video_id, telegram_user_headers(USER_UID))["token"]
    head, payload, signature = token.split(".")
    tampered = f"{head}.{payload}.A{signature}"

    resp = client.get(
        f"{API}/dataset/cards",
        params={"video_id": video_id},
        headers={"Authorization": f"Bearer {tampered}"},
    )
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"]["code"] == "unauthorized"


def test_expired_portal_token_is_rejected(client):
    """An expired token is a 401, not a wider scope."""
    video_id = _make_video()
    token, _ = create_access_token(
        f"guest:{uuid4().hex[:12]}",
        role="guest",
        video_ids=[video_id],
        video_roles={video_id: "viewer"},
        ttl_minutes=-1,
    )

    resp = client.get(
        f"{API}/dataset/cards",
        params={"video_id": video_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"]["code"] == "unauthorized"


def test_portal_link_missing_video_is_404(client):
    """A video that does not exist answers the standard error envelope."""
    resp = client.post(
        f"{API}/videos/{uuid4()}/portal-link",
        headers=telegram_user_headers(USER_UID),
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"


# --------------------------------------------------------------- who may mint


def test_telegram_user_mints_own_portal_link(client):
    """The bot calls with the service token plus `X-Requested-By`; the user
    whose `telegram:<uid>` owns the row gets that unit's link."""
    video_id = _make_video(created_by=f"telegram:{USER_UID}")
    body = _mint(client, video_id, telegram_user_headers(USER_UID))

    payload = decode_token(body["token"])
    assert payload["videos"] == [video_id]
    assert body["path"] == f"/video/{video_id}?t={body['token']}"


def test_telegram_user_cannot_mint_another_users_portal_link(client):
    """Regression blocker: the caller names someone else's UUID.

    The bot token is valid and the header names a real Telegram user, but the
    row belongs to a different `telegram:<uid>`. That is a 403, not a link.
    `video_role()` answers "editor" for every video whenever the role is
    `telegram`, so a guard built on it would hand out any unit here.
    """
    mine = _make_video(created_by=f"telegram:{USER_UID}")
    theirs = _make_video(created_by="telegram:999999")

    # Exactly what the bot sends: service token + the acting user's id.
    resp = client.post(
        f"{API}/videos/{theirs}/portal-link",
        headers=telegram_user_headers(USER_UID),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"

    # The same credentials still mint the caller's own unit.
    assert _mint(client, mine, telegram_user_headers(USER_UID))["token"]


def _link(uid: int, role: str) -> str:
    """Bind `uid` to an active console account of `role`, as /link does.

    `AdminUser.telegram_user_id` is the link field, and `_resolve_principal`
    re-reads the row per request: the actor stays `telegram:<uid>` while the
    role becomes the account's.
    """
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


def test_linked_ordinary_user_mints_own_portal_link(client):
    """Regression: a uid linked to an active `user` account has `role="user"`,
    not `telegram`, so the old role check refused its own units. Ownership
    keys on the actor (`telegram:<uid>`), which a link does not change."""
    uid = 770001
    _link(uid, "user")
    video_id = _make_video(created_by=f"telegram:{uid}")

    body = _mint(client, video_id, telegram_user_headers(uid))

    payload = decode_token(body["token"])
    assert payload["videos"] == [video_id]
    assert body["path"] == f"/video/{video_id}?t={body['token']}"


def test_linked_ordinary_user_cannot_mint_another_users_portal_link(client):
    """The same linked user naming someone else's row is still a 403: the
    actor comparison, not the widened role, is what admits."""
    uid = 770002
    _link(uid, "user")
    theirs = _make_video(created_by="telegram:770099")

    resp = client.post(
        f"{API}/videos/{theirs}/portal-link",
        headers=telegram_user_headers(uid),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"


def test_raw_service_token_without_requested_by_is_refused(client, bot_headers):
    """The bare service principal (`telegram:bot`, no `X-Requested-By`) is
    refused: every mint must name the end user, so a handler that forgets the
    header fails closed instead of handing out arbitrary links."""
    video_id = _make_video(created_by=f"telegram:{USER_UID}")
    resp = client.post(
        f"{API}/videos/{video_id}/portal-link", headers=bot_headers
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"


def test_portal_link_anonymous_gets_401(client):
    """No credential is the usual `401`, before any ownership decision."""
    video_id = _make_video()
    resp = client.post(f"{API}/videos/{video_id}/portal-link")
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"]["code"] == "unauthorized"


def test_guest_token_cannot_mint_portal_link(client):
    """A portal guest is a reader, never an issuer."""
    video_id = _make_video()
    guest_token, _ = issue_guest_token()
    resp = client.post(
        f"{API}/videos/{video_id}/portal-link",
        headers={"Authorization": f"Bearer {guest_token}"},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"


def test_admin_can_mint_portal_link(client, auth_headers):
    """The operator surface may hand out the same link for any existing unit."""
    video_id = _make_video()
    body = _mint(client, video_id, auth_headers)
    assert body["path"].endswith(f"/video/{video_id}?t={body['token']}")
