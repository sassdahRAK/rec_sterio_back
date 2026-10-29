"""Shared builders for the dataset-surface tests.

The folder/share/passcode/public routers all hang off the same objects, so the
tests build them the same way: a real upload (which writes a media file) plus a
real transcript version. Tokens are issued through the production helpers
rather than forged, so a test cannot pass by bypassing the signing path.
"""
from __future__ import annotations

from uuid import uuid4

from app.auth import create_access_token, issue_folder_token, issue_guest_token


def media_bytes() -> bytes:
    """A real file payload, unique per call.

    Ingestion dedupes on sha256, so a constant payload would make every upload
    after the first a 409.
    """
    return b"\x00\x00\x00\x18ftypmp42" + uuid4().hex.encode() * 16


# Kept for tests that assert on the raw payload.
MEDIA_BYTES = media_bytes()


def user_email(tag: str = "user") -> str:
    return f"{tag}-{abs(hash(tag)) % 10**8}@test.local"


def ensure_user(tag: str = "user") -> str:
    """Create the account row a user token refers to, and return its email.

    A user token is only accepted when the account exists, and a plain user
    cannot log in with a password (no password is set), so tests that need one
    create the row directly. `tag` names the account (`ensure_user("viewer")`);
    it is not an address.
    """
    from app.db import SessionLocal
    from app.models import AdminUser
    from sqlalchemy import select

    email = user_email(tag)
    with SessionLocal() as db:
        user = db.execute(select(AdminUser).where(AdminUser.email == email)).scalar_one_or_none()
        if user is None:
            db.add(AdminUser(email=email, role="user", password_hash=None))
            db.commit()
    return email


def user_headers(email: str | None = None) -> dict:
    """A signed-in dataset user (not the operator)."""
    email = email or user_email()
    token, _ = create_access_token(email, "user")
    return {"Authorization": f"Bearer {token}"}


def guest_headers() -> dict:
    """An anonymous visitor: public folders only."""
    token, _ = issue_guest_token()
    return {"Authorization": f"Bearer {token}"}


def bot_headers() -> dict:
    """The Telegram bot's service identity.

    The API treats this principal as an editor on every folder, so it is the
    credential the dataset commands run under.
    """
    from app.config import settings

    return {
        "Authorization": f"Bearer {settings.bot_api_token}",
        "X-Requested-By": "telegram:bot",
    }


def telegram_user_headers(uid: int) -> dict:
    """A Telegram user acting through the bot: service token + own id.

    Both credentials are what the /bot-users routes (and the consent gate)
    require — the token alone is the bot's identity, not the user's.
    """
    from app.config import settings

    return {
        "Authorization": f"Bearer {settings.bot_api_token}",
        "X-Requested-By": f"telegram:{uid}",
    }


def scoped_headers(folder_id, role: str = "viewer") -> dict:
    """What a passcode or share redemption hands back."""
    token, _ = issue_folder_token(folder_id, role)
    return {"Authorization": f"Bearer {token}"}


def upload_video(client, headers, *, folder_id=None, title="Clip", start_transcription=False):
    """Ingest a real media file, so the export has something to archive."""
    data = {"title": title, "start_transcription": str(start_transcription).lower()}
    if folder_id is not None:
        data["folder_id"] = str(folder_id)
    resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("clip.mp4", media_bytes(), "video/mp4")},
        data=data,
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def make_transcript(client, headers, video_id, content, cleaned=None):
    payload = {"content": content}
    if cleaned is not None:
        payload["cleaned"] = cleaned
    resp = client.put(
        f"/api/v1/videos/{video_id}/transcript", json=payload, headers=headers
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def create_folder(client, headers, *, name="Dataset", visibility="private", parent_id=None,
                  position=None):
    payload = {"name": name, "visibility": visibility}
    if parent_id is not None:
        payload["parent_id"] = str(parent_id)
    if position is not None:
        payload["position"] = position
    resp = client.post("/api/v1/folders", json=payload, headers=headers)
    return resp


def make_folder(client, headers, *, name="Dataset", visibility="private", parent_id=None):
    resp = create_folder(
        client, headers, name=name, visibility=visibility, parent_id=parent_id
    )
    assert resp.status_code == 201, resp.text
    return resp.json()
