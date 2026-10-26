"""Folders, membership, share links, passcodes, dataset cards and ZIP export.

Covers instruction.txt #1 (folders + sharing + roles), #3 (guest reads public
data), #4/#7 (passcode door codes), #5/#6 (ZIP downloads), #8/#9 (cards with a
Raw/Cleaned badge, editable by people with the right role).
"""
from __future__ import annotations

import io
import zipfile

from sqlalchemy import select

from app.auth import create_access_token, hash_password
from app.db import SessionLocal
from app.models import AdminUser, Folder, Transcript, Video
from app.services import storage
from tests.conftest import ADMIN_EMAIL

API = "/api/v1"


# ------------------------------------------------------------------ helpers


def make_folder(client, headers, name="Dataset", visibility="private", parent_id=None):
    resp = client.post(
        f"{API}/folders",
        json={
            "name": name,
            "visibility": visibility,
            "parent_id": parent_id,
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def make_member(email="member@test.local", password="member-password"):
    """Create a member account and return bearer headers for it."""
    with SessionLocal() as db:
        user = db.execute(
            select(AdminUser).where(AdminUser.email == email)
        ).scalar_one_or_none()
        if user is None:
            user = AdminUser(
                email=email,
                password_hash=hash_password(password),
                role="member",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        user_id = str(user.id)
    token, _ = create_access_token(email, role="member")
    return {"Authorization": f"Bearer {token}"}, user_id


def make_video(folder_id, title="Clip", cleaned=False, with_files=True):
    """A ready video with media, audio and a transcript — enough for a ZIP."""
    with SessionLocal() as db:
        video = Video(
            source_url=f"upload://{title}.mp4",
            platform="direct",
            title=title,
            status="transcribed",
            progress=100,
            folder_id=folder_id,
            created_by=f"admin:{ADMIN_EMAIL}",
        )
        db.add(video)
        db.flush()
        db.add(
            Transcript(
                video_id=video.id,
                version=1,
                source="whisper",
                language="km",
                content="[km] សួស្តី [en] hello world",
                segments=[
                    {"start": 0.0, "end": 1.2, "text": "សួស្តី", "language": "km"},
                    {"start": 1.2, "end": 2.4, "text": "hello world", "language": "en"},
                ],
                status="ready",
                cleaned=cleaned,
                created_by="system",
            )
        )
        db.commit()
        video_id = str(video.id)

    if with_files:
        storage.media_file(video_id, "mp4").write_bytes(b"fake-video-bytes")
        storage.audio_file(video_id).write_bytes(b"fake-audio-bytes")
    return video_id


def guest_headers(client):
    resp = client.post(f"{API}/auth/guest")
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['token']}"}


# ------------------------------------------------------------------ folders


def test_admin_can_create_public_and_private_folders(client, auth_headers):
    public = make_folder(client, auth_headers, "Open set", "public")
    private = make_folder(client, auth_headers, "Internal", "private")
    assert public["visibility"] == "public"
    assert public["my_role"] == "editor"
    assert private["visibility"] == "private"

    listing = client.get(f"{API}/folders", headers=auth_headers).json()
    names = {item["name"] for item in listing["items"]}
    assert {"Open set", "Internal"} <= names


def test_subfolders_nest_and_count_their_videos(client, auth_headers):
    root = make_folder(client, auth_headers, "Superset", "public")
    child = make_folder(client, auth_headers, "Unit A", "public", parent_id=root["id"])
    make_video(child["id"])

    detail = client.get(f"{API}/folders/{root['id']}", headers=auth_headers).json()
    assert [c["name"] for c in detail["children"]] == ["Unit A"]
    assert detail["children"][0]["video_count"] == 1


def test_member_cannot_create_a_public_folder(client, auth_headers):
    headers, _ = make_member()
    resp = client.post(
        f"{API}/folders", json={"name": "Nope", "visibility": "public"}, headers=headers
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"

    own = client.post(
        f"{API}/folders", json={"name": "Mine", "visibility": "private"}, headers=headers
    )
    assert own.status_code == 201
    assert own.json()["my_role"] == "editor"


def test_guest_sees_public_tree_but_not_private_folders(client, auth_headers):
    public_root = make_folder(client, auth_headers, "Public root", "public")
    make_folder(client, auth_headers, "Public child", "public", parent_id=public_root["id"])
    private = make_folder(client, auth_headers, "Secret", "private")

    guest = guest_headers(client)
    tree = client.get(f"{API}/public/folders", headers=guest).json()
    assert [item["name"] for item in tree["items"]] == ["Public root"]
    assert [c["name"] for c in tree["items"][0]["children"]] == ["Public child"]

    hidden = client.get(f"{API}/public/folders/{private['id']}", headers=guest)
    assert hidden.status_code == 404
    assert hidden.json()["error"]["code"] == "not_found"


def test_anonymous_without_token_can_read_public_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, "Open", "public")
    resp = client.get(f"{API}/public/folders/{folder['id']}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Open"


# ------------------------------------------------------------------- roles


def test_member_roles_gate_editing(client, auth_headers):
    folder = make_folder(client, auth_headers, "Shared", "private")
    video_id = make_video(folder["id"])
    headers, user_id = make_member()

    # Not invited yet: the folder is invisible, not merely read-only.
    assert client.get(f"{API}/folders/{folder['id']}", headers=headers).status_code == 404

    invited = client.post(
        f"{API}/folders/{folder['id']}/members",
        json={"email": "member@test.local", "role": "viewer"},
        headers=auth_headers,
    )
    assert invited.status_code == 201, invited.text

    # Viewer: can read, cannot edit the transcript or the folder.
    assert client.get(f"{API}/folders/{folder['id']}", headers=headers).status_code == 200
    denied = client.put(
        f"{API}/videos/{video_id}/transcript",
        json={"content": "changed"},
        headers=headers,
    )
    assert denied.status_code == 403
    assert client.patch(
        f"{API}/folders/{folder['id']}", json={"name": "Renamed"}, headers=headers
    ).status_code == 403

    # Commenter can comment but still not edit.
    client.patch(
        f"{API}/folders/{folder['id']}/members/{user_id}",
        json={"role": "commenter"},
        headers=auth_headers,
    )
    comment = client.post(
        f"{API}/folders/{folder['id']}/comments",
        json={"body": "nice set"},
        headers=headers,
    )
    assert comment.status_code == 201
    assert client.put(
        f"{API}/videos/{video_id}/transcript",
        json={"content": "changed"},
        headers=headers,
    ).status_code == 403

    # Editor can edit and delete cards (#9).
    client.patch(
        f"{API}/folders/{folder['id']}/members/{user_id}",
        json={"role": "editor"},
        headers=auth_headers,
    )
    edited = client.put(
        f"{API}/videos/{video_id}/transcript",
        json={"content": "human corrected"},
        headers=headers,
    )
    assert edited.status_code == 201
    assert edited.json()["cleaned"] is True


# --------------------------------------------------------------- share links


def test_share_link_grants_scoped_read_and_can_be_revoked(client, auth_headers):
    folder = make_folder(client, auth_headers, "Shared set", "private")
    video_id = make_video(folder["id"])

    created = client.post(
        f"{API}/folders/{folder['id']}/shares",
        json={"role": "viewer"},
        headers=auth_headers,
    )
    assert created.status_code == 201, created.text
    share = created.json()
    assert share["url"].endswith(f"/share/{share['token']}")

    redeemed = client.post(f"{API}/public/shares/{share['token']}/redeem", json={})
    assert redeemed.status_code == 200
    token = redeemed.json()["token"]
    scoped = {"Authorization": f"Bearer {token}"}

    folder_view = client.get(f"{API}/public/folders/{folder['id']}", headers=scoped)
    assert folder_view.status_code == 200
    assert folder_view.json()["cards"][0]["video_id"] == video_id
    # The scoped guest may read the transcript but not edit it.
    assert client.get(f"{API}/videos/{video_id}/transcript", headers=scoped).status_code == 200
    assert client.put(
        f"{API}/videos/{video_id}/transcript",
        json={"content": "nope"},
        headers=scoped,
    ).status_code == 403

    client.delete(f"{API}/shares/{share['id']}", headers=auth_headers)
    gone = client.post(f"{API}/public/shares/{share['token']}/redeem", json={})
    assert gone.status_code == 404


def test_share_link_with_unknown_token_is_404(client):
    resp = client.post(f"{API}/public/shares/not-a-token/redeem", json={})
    assert resp.status_code == 404


# ----------------------------------------------------------------- passcodes


def test_passcode_is_generated_returned_once_and_opens_the_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, "Locked", "private")
    created = client.post(
        f"{API}/folders/{folder['id']}/passcodes",
        json={"label": "for guests"},
        headers=auth_headers,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["code"], "a generated code must be returned once so it can be shared"
    assert len(body["code"]) == 9 and body["code"][4] == "-"

    # Listing never leaks the code or its hash.
    listing = client.get(f"{API}/folders/{folder['id']}/passcodes", headers=auth_headers).json()
    assert listing[0]["code"] is None
    assert "code_hash" not in listing[0]

    opened = client.post(
        f"{API}/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": body["code"]},
    )
    assert opened.status_code == 200
    assert opened.json()["access"] == "passcode"
    scoped = {"Authorization": f"Bearer {opened.json()['token']}"}
    assert client.get(f"{API}/public/folders/{folder['id']}", headers=scoped).status_code == 200


def test_wrong_passcode_is_401(client, auth_headers):
    folder = make_folder(client, auth_headers, "Locked", "private")
    client.post(
        f"{API}/folders/{folder['id']}/passcodes",
        json={"code": "correct-horse"},
        headers=auth_headers,
    )
    bad = client.post(
        f"{API}/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "wrong"},
    )
    assert bad.status_code == 401


def test_one_time_passcode_burns_after_first_use(client, auth_headers):
    folder = make_folder(client, auth_headers, "One shot", "private")
    client.post(
        f"{API}/folders/{folder['id']}/passcodes",
        json={"code": "single-use", "one_time": True},
        headers=auth_headers,
    )
    first = client.post(
        f"{API}/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "single-use"},
    )
    assert first.status_code == 200
    second = client.post(
        f"{API}/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "single-use"},
    )
    assert second.status_code == 401


def test_expired_passcode_is_rejected(client, auth_headers):
    folder = make_folder(client, auth_headers, "Old code", "private")
    client.post(
        f"{API}/folders/{folder['id']}/passcodes",
        json={"code": "stale-code", "expires_at": "2020-01-01T00:00:00Z"},
        headers=auth_headers,
    )
    resp = client.post(
        f"{API}/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "stale-code"},
    )
    assert resp.status_code == 401


def test_device_limit_and_repeat_device(client, auth_headers):
    folder = make_folder(client, auth_headers, "Two devices", "private")
    created = client.post(
        f"{API}/folders/{folder['id']}/passcodes",
        json={"code": "two-devices", "max_devices": 2},
        headers=auth_headers,
    ).json()

    def redeem(device):
        return client.post(
            f"{API}/public/passcodes/redeem",
            json={"folder_id": folder["id"], "code": "two-devices", "device_key": device},
        )

    assert redeem("laptop").status_code == 200
    assert redeem("laptop").status_code == 200  # a known device is not charged twice
    assert redeem("phone").status_code == 200
    third = redeem("tablet")
    assert third.status_code == 401
    assert "device limit" in third.json()["error"]["message"]

    listing = client.get(
        f"{API}/folders/{folder['id']}/passcodes", headers=auth_headers
    ).json()
    assert listing[0]["device_count"] == 2
    assert listing[0]["id"] == created["id"]


# ------------------------------------------------------------- dataset cards


def test_dataset_cards_show_badge_preview_and_only_visible_folders(client, auth_headers):
    public = make_folder(client, auth_headers, "Public", "public")
    private = make_folder(client, auth_headers, "Private", "private")
    raw_id = make_video(public["id"], title="Raw clip")
    clean_id = make_video(private["id"], title="Clean clip", cleaned=True)

    guest = guest_headers(client)
    cards = client.get(f"{API}/dataset/cards", headers=guest).json()
    assert [c["video_id"] for c in cards["items"]] == [raw_id]
    assert cards["items"][0]["badge"] == "raw"
    assert cards["items"][0]["transcript_preview"]

    admin_cards = client.get(f"{API}/dataset/cards", headers=auth_headers).json()
    by_id = {c["video_id"]: c for c in admin_cards["items"]}
    assert by_id[raw_id]["badge"] == "raw"
    assert by_id[clean_id]["badge"] == "cleaned"

    cleaned_only = client.get(
        f"{API}/dataset/cards", params={"cleaned": "true"}, headers=auth_headers
    ).json()
    assert [c["video_id"] for c in cleaned_only["items"]] == [clean_id]


def test_badge_toggles_without_creating_a_transcript_version(client, auth_headers):
    folder = make_folder(client, auth_headers, "Folder", "private")
    video_id = make_video(folder["id"])

    before = client.get(f"{API}/videos/{video_id}/transcript", headers=auth_headers).json()
    patched = client.patch(
        f"{API}/videos/{video_id}/transcript",
        json={"cleaned": True},
        headers=auth_headers,
    )
    assert patched.status_code == 200
    assert patched.json()["cleaned"] is True
    after = client.get(f"{API}/videos/{video_id}/transcript", headers=auth_headers).json()
    assert after["id"] == before["id"]
    assert after["version"] == before["version"]


# ------------------------------------------------------------- media access


def test_public_video_media_is_readable_by_guest_but_private_is_not(client, auth_headers):
    public = make_folder(client, auth_headers, "Open", "public")
    public_video = make_video(public["id"])
    private = make_folder(client, auth_headers, "Closed", "private")
    private_video = make_video(private["id"])

    guest = guest_headers(client)
    ok = client.get(f"{API}/videos/{public_video}/media", headers=guest)
    assert ok.status_code == 200
    assert ok.content == b"fake-video-bytes"

    denied = client.get(f"{API}/videos/{private_video}/media", headers=guest)
    assert denied.status_code == 404


def test_unfiled_video_is_console_only(client, auth_headers):
    with SessionLocal() as db:
        video = Video(
            source_url="upload://unfiled.mp4",
            platform="direct",
            status="transcribed",
            created_by=f"admin:{ADMIN_EMAIL}",
        )
        db.add(video)
        db.commit()
        video_id = str(video.id)

    guest = guest_headers(client)
    assert client.get(f"{API}/videos/{video_id}", headers=guest).status_code == 403
    assert client.get(f"{API}/videos/{video_id}", headers=auth_headers).status_code == 200


# ---------------------------------------------------------------- zip export


def test_folder_zip_contains_video_audio_and_transcript_per_unit(client, auth_headers):
    root = make_folder(client, auth_headers, "Superset set", "public")
    unit = make_folder(client, auth_headers, "Unit One", "public", parent_id=root["id"])
    make_video(unit["id"], title="Khmer clip")

    resp = client.get(f"{API}/folders/{root['id']}/download", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        joined = "\n".join(names)
        assert "Superset set/Unit One/" in joined
        assert "/video/" in joined and "/audio/" in joined and "/transcript/" in joined
        transcript_entry = next(n for n in names if "/transcript/" in n)
        text = zf.read(transcript_entry).decode("utf-8")
        assert "[km] សួស្តី" in text and "[en] hello world" in text


def test_guest_can_download_a_public_folder_zip(client, auth_headers):
    folder = make_folder(client, auth_headers, "Open set", "public")
    make_video(folder["id"])

    guest = guest_headers(client)
    resp = client.get(f"{API}/folders/{folder['id']}/download", headers=guest)
    assert resp.status_code == 200
    assert resp.content[:2] == b"PK"


def test_zip_of_an_empty_folder_is_409(client, auth_headers):
    folder = make_folder(client, auth_headers, "Empty", "private")
    resp = client.get(f"{API}/folders/{folder['id']}/download", headers=auth_headers)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "nothing_to_export"


# ------------------------------------------------------------------- extras


def test_me_reports_the_caller_and_scopes(client, auth_headers):
    me = client.get(f"{API}/auth/me", headers=auth_headers)
    assert me.status_code == 200
    assert me.json()["email"] == ADMIN_EMAIL
    assert me.json()["role"] == "admin"

    guest = client.post(f"{API}/auth/guest").json()
    assert guest["role"] == "guest"
    guest_me = client.get(
        f"{API}/auth/me", headers={"Authorization": f"Bearer {guest['token']}"}
    )
    assert guest_me.status_code == 200
    assert guest_me.json()["role"] == "guest"


def test_google_sign_in_is_disabled_without_a_client_id(client, auth_headers):
    from app.config import settings

    original = settings.google_client_id
    settings.google_client_id = None
    try:
        resp = client.post(f"{API}/auth/google", json={"credential": "x"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "google_not_configured"
    finally:
        settings.google_client_id = original


def test_public_config_advertises_google_only_when_configured(client):
    from app.config import settings

    original = settings.google_client_id
    try:
        settings.google_client_id = None
        body = client.get(f"{API}/public/config").json()
        assert body["google_enabled"] is False
        settings.google_client_id = "abc.apps.googleusercontent.com"
        body = client.get(f"{API}/public/config").json()
        assert body["google_enabled"] is True
        assert body["google_client_id"] == "abc.apps.googleusercontent.com"
    finally:
        settings.google_client_id = original
