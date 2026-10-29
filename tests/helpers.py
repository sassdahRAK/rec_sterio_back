"""Shared builders for the dataset-surface tests.

The folder/share/passcode/public routers all hang off the same objects, so the
tests build them the same way: a real upload (which writes a media file) plus a
real transcript version. Tokens are issued through the production helpers
rather than forged, so a test cannot pass by bypassing the signing path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from app.auth import create_access_token, issue_folder_token, issue_guest_token


@lru_cache(maxsize=None)
def real_container_prefix(suffix: str) -> bytes:
    """The first bytes of a genuine container, built once per suffix by ffmpeg.

    The upload route's ffprobe gate refuses bytes that no decoder can parse,
    so a fixture that fakes only the magic header would now be rejected. A
    short real encoding (0.2 s, tiny) is made once per suffix and truncated
    after its moov/metadata boxes: faststart puts them at the front, so the
    truncated prefix still probes cleanly and the unique filler below keeps
    every payload's sha256 distinct (uploads dedupe on the checksum).
    Returns b'' when ffmpeg is unavailable; the synthetic fallback applies and
    the (absent) ffprobe gate stays inert.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return b""
    encoders = {
        ".mp4": ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.2",
                 "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=0.2",
                 "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "24k", "-movflags", "+faststart"],
        ".mp3": ["-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=0.2"],
    }
    args = encoders.get(suffix)
    if args is None:
        return b""
    fd, path = tempfile.mkstemp(prefix="recsterio-fixture-", suffix=suffix)
    os.close(fd)
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", *args, "-y", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False,
        )
        if proc.returncode != 0:
            return b""
        keep = 4096 if suffix == ".mp4" else 1024
        with open(path, "rb") as f:
            return f.read(keep)
    finally:
        os.unlink(path)


def media_bytes(suffix: str = ".mp4") -> bytes:
    """A real container payload, unique per call.

    Ingestion dedupes on sha256, so a constant payload would make every upload
    after the first a 409 — the unique filler appended after the container's
    leading boxes takes care of that (the same trick makes each uploaded file
    a fresh checksum while the bytes stay a decodable container).
    """
    prefix = real_container_prefix(suffix)
    filler = uuid4().hex.encode() * 16
    if prefix:
        return prefix + filler
    # No ffmpeg: fake header bytes (no ffprobe to catch them, either).
    magics = {".mp4": b"\x00\x00\x00\x18ftypmp42", ".mp3": b"ID3\x00\x07"}
    return magics.get(suffix, b"ftyp-fallback") + filler


# Kept for tests that assert on the raw payload.
MEDIA_BYTES = media_bytes()


@lru_cache(maxsize=1)
def large_media_file(seconds: int = 40, video_kbps: int = 12_000):
    """A genuinely large 720p testsrc2 mp4 (>50 MB), encoded once per process.

    The delivery-compression tests need an original above the 50 MB delivery
    threshold; a fixture that size is far too big to build per test, so it is
    encoded to a fixed path in the temp dir keyed by its spec and reused while
    the file is present. Returns None when ffmpeg is unavailable, and the
    caller skips. Complex testsrc2 motion keeps the CRF ladders honest.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    path = Path(tempfile.gettempdir()) / f"recsterio-large-fixture-{seconds}s-{video_kbps}k.mp4"
    if not path.exists() or path.stat().st_size < 50 * 1024 * 1024:
        proc = subprocess.run(
            [
                ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}",
                "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:duration={seconds}",
                "-c:v", "libx264", "-preset", "veryfast", "-b:v", f"{video_kbps}k",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                str(path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=False,
        )
        if proc.returncode != 0 or path.stat().st_size < 50 * 1024 * 1024:
            return None
    return path


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
