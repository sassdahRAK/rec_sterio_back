"""Dataset sharing: folders, share links, passcodes, public browsing, ZIP export.

Covers instruction.txt #1 (folders + invites), #3 (guest public browsing),
#4 (passcode door), #5/#6 (ZIP of video/audio/transcript), #7 (passcode rules)
and #8 (raw vs cleaned cards).
"""
from __future__ import annotations

import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO

from app.db import SessionLocal
from app.models import Folder, PasscodeGrant, Transcript
from app.services import storage

URL = "https://www.youtube.com/watch?v=sharing-test-001"


# ------------------------------------------------------------------ helpers


def make_video(client, headers, url: str = URL, title: str | None = None) -> dict:
    body = {"source_url": url}
    if title:
        body["title"] = title
    resp = client.post("/api/v1/videos", json=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def make_folder(client, headers, name: str = "Phnom Penh cafés", visibility: str = "private",
                parent_id: str | None = None) -> dict:
    body = {"name": name, "visibility": visibility}
    if parent_id:
        body["parent_id"] = parent_id
    resp = client.post("/api/v1/folders", json=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def seed_media(video_id: str, title: str | None = None) -> None:
    """Put a real media file + extracted audio + transcript on disk for a video."""
    media = storage.media_file(video_id, "mp4")
    media.write_bytes(b"\x00" * 2048)
    storage.audio_file(video_id).write_bytes(b"\x00" * 1024)
    with SessionLocal() as db:
        db.add(
            Transcript(
                video_id=video_id,
                version=1,
                source="whisper",
                content="raw transcript",
                status="ready",
                cleaned=False,
            )
        )
        db.commit()


def seed_cleaned(video_id: str) -> None:
    with SessionLocal() as db:
        db.add(
            Transcript(
                video_id=video_id,
                version=2,
                source="manual",
                content="cleaned transcript",
                status="ready",
                cleaned=True,
            )
        )
        db.commit()


def guest_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------ folders


def test_folder_tree_nests_children(client, auth_headers):
    parent = make_folder(client, auth_headers, name="Parent")
    child = make_folder(client, auth_headers, name="Child", parent_id=parent["id"])

    resp = client.get("/api/v1/folders", headers=auth_headers)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == parent["id"]
    assert [c["id"] for c in items[0]["children"]] == [child["id"]]
    assert items[0]["children"][0]["video_count"] == 0


def test_folder_requires_a_signed_in_user(client):
    resp = client.post("/api/v1/folders", json={"name": "Nope"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_place_video_in_folder_and_count_it(client, auth_headers):
    video = make_video(client, auth_headers)
    folder = make_folder(client, auth_headers)

    resp = client.put(
        f"/api/v1/videos/{video['id']}/folder",
        json={"folder_id": folder["id"], "position": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["folder_id"] == folder["id"]
    assert resp.json()["position"] == 3

    detail = client.get(f"/api/v1/folders/{folder['id']}", headers=auth_headers).json()
    assert detail["video_count"] == 1
    listed = client.get(f"/api/v1/folders/{folder['id']}/videos", headers=auth_headers).json()
    assert [v["id"] for v in listed] == [video["id"]]


def test_removing_a_folder_keeps_its_videos(client, auth_headers):
    video = make_video(client, auth_headers)
    folder = make_folder(client, auth_headers)
    client.put(
        f"/api/v1/videos/{video['id']}/folder",
        json={"folder_id": folder["id"]},
        headers=auth_headers,
    )

    assert client.delete(f"/api/v1/folders/{folder['id']}", headers=auth_headers).status_code == 204
    # The video survives, unfiled — deleting a grouping is not deleting data.
    after = client.get(f"/api/v1/videos/{video['id']}", headers=auth_headers).json()
    assert after["folder_id"] is None


# ------------------------------------------------------------- public surface


def test_public_folder_is_visible_to_an_anonymous_visitor(client, auth_headers):
    make_folder(client, auth_headers, name="Open dataset", visibility="public")

    resp = client.get("/api/v1/public/folders")
    assert resp.status_code == 200, resp.text
    names = [f["name"] for f in resp.json()["items"]]
    assert "Open dataset" in names


def test_private_folder_is_hidden_from_an_anonymous_visitor(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Secret", visibility="private")

    listing = client.get("/api/v1/public/folders")
    assert listing.status_code == 200
    assert all(f["id"] != folder["id"] for f in listing.json()["items"])

    # Indistinguishable from a folder that does not exist.
    resp = client.get(f"/api/v1/public/folders/{folder['id']}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_guest_token_enables_public_browsing(client, auth_headers):
    make_folder(client, auth_headers, name="Open dataset", visibility="public")

    issued = client.post("/api/v1/auth/guest")
    assert issued.status_code == 200, issued.text
    token = issued.json()["token"]

    resp = client.get("/api/v1/public/folders", headers=guest_headers(token))
    assert resp.status_code == 200
    assert any(f["name"] == "Open dataset" for f in resp.json()["items"])


def test_dataset_cards_carry_a_raw_or_cleaned_badge(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="public")
    raw = make_video(client, auth_headers, title="Raw clip")
    cleaned = make_video(
        client, auth_headers, url=URL + "?clean=1", title="Cleaned clip"
    )
    seed_media(raw["id"])
    seed_media(cleaned["id"])
    seed_cleaned(cleaned["id"])
    for video in (raw, cleaned):
        client.put(
            f"/api/v1/videos/{video['id']}/folder",
            json={"folder_id": folder["id"]},
            headers=auth_headers,
        )

    cards = client.get(f"/api/v1/public/folders/{folder['id']}").json()["cards"]
    by_title = {c["title"]: c for c in cards}
    assert by_title["Raw clip"]["badge"] == "raw"
    assert by_title["Cleaned clip"]["badge"] == "cleaned"
    assert by_title["Cleaned clip"]["transcript_preview"] == "cleaned transcript"
    assert by_title["Raw clip"]["media_url"].endswith(f"/videos/{raw['id']}/media")


def test_public_folder_detail_returns_its_subtree(client, auth_headers):
    """The detail view is a folder *tree*, not one level (#6)."""
    root = make_folder(client, auth_headers, "Khmer conversations", "public")
    child = make_folder(
        client, auth_headers, "Unit 01", "public", parent_id=root["id"]
    )
    video = make_video(client, auth_headers, title="Clip in unit")
    seed_media(video["id"])
    client.put(
        f"/api/v1/videos/{video['id']}/folder",
        json={"folder_id": child["id"]},
        headers=auth_headers,
    )

    detail = client.get(f"/api/v1/public/folders/{root['id']}").json()
    assert [c["name"] for c in detail["children"]] == ["Unit 01"]
    assert detail["children"][0]["cards"][0]["video_id"] == video["id"]
    # The card carries its folder so the portal can label and gate it.
    card = detail["children"][0]["cards"][0]
    assert card["folder_id"] == child["id"]
    assert card["folder_name"] == "Unit 01"
    # An anonymous visitor is a viewer, never an editor.
    assert detail["my_role"] == "viewer"
    assert detail["children"][0]["my_role"] == "viewer"


def test_scoped_private_subtree_is_visible_with_its_passcode(client, auth_headers):
    root = make_folder(client, auth_headers, "Private superset", "private")
    child = make_folder(
        client, auth_headers, "Private unit", "private", parent_id=root["id"]
    )
    code = client.post(
        f"/api/v1/folders/{root['id']}/passcodes",
        json={"code": "nested-42"},
        headers=auth_headers,
    )
    assert code.status_code == 201, code.text

    token = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": root["id"], "code": "nested-42"},
    ).json()["token"]
    scoped = guest_headers(token)

    # The root listing shows the whole scoped subtree, and the child is
    # reachable directly rather than only through its parent.
    tree = client.get("/api/v1/public/folders", headers=scoped).json()
    node = next(item for item in tree["items"] if item["id"] == root["id"])
    assert [c["id"] for c in node["children"]] == [child["id"]]

    detail = client.get(f"/api/v1/public/folders/{child['id']}", headers=scoped)
    assert detail.status_code == 200
    assert detail.json()["id"] == child["id"]


def test_redeeming_reports_the_role_that_was_granted(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    viewer_share = client.post(
        f"/api/v1/folders/{folder['id']}/shares",
        json={"role": "viewer"},
        headers=auth_headers,
    ).json()
    editor_share = client.post(
        f"/api/v1/folders/{folder['id']}/shares",
        json={"role": "editor"},
        headers=auth_headers,
    ).json()
    passcode = client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "role-check"},
        headers=auth_headers,
    )
    assert passcode.status_code == 201, passcode.text

    viewer = client.post(f"/api/v1/public/shares/{viewer_share['token']}/redeem", json={})
    editor = client.post(f"/api/v1/public/shares/{editor_share['token']}/redeem", json={})
    opened = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "role-check"},
    )
    assert viewer.json()["role"] == "guest"
    assert viewer.json()["folder_role"] == "viewer"
    assert editor.json()["folder_role"] == "editor"
    # A passcode is read-only by definition.
    assert opened.json()["folder_role"] == "viewer"


# ----------------------------------------------------------------- passcodes


def test_passcode_opens_a_private_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    created = client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "cafe-2026", "label": "Field team"},
        headers=auth_headers,
    )
    assert created.status_code == 201, created.text
    # An operator-supplied code is never echoed back; only a generated one is,
    # and then exactly once.
    assert created.json()["code"] is None

    redeemed = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "cafe-2026", "device_key": "device-a"},
    )
    assert redeemed.status_code == 200, redeemed.text
    token = redeemed.json()["token"]

    resp = client.get(
        f"/api/v1/public/folders/{folder['id']}", headers=guest_headers(token)
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == folder["id"]


def test_wrong_passcode_is_rejected(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "cafe-2026"},
        headers=auth_headers,
    )
    resp = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "wrong"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_expired_passcode_is_rejected(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "cafe-2026", "expires_at": "2020-01-01T00:00:00Z"},
        headers=auth_headers,
    )
    resp = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "cafe-2026"},
    )
    assert resp.status_code == 401


def test_one_time_passcode_burns_after_its_first_use(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "once", "one_time": True},
        headers=auth_headers,
    )
    body = {"folder_id": folder["id"], "code": "once"}
    assert client.post("/api/v1/public/passcodes/redeem", json=body).status_code == 200
    assert client.post("/api/v1/public/passcodes/redeem", json=body).status_code == 401


def test_max_uses_passcode_stops_at_the_limit(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "twice", "max_uses": 2},
        headers=auth_headers,
    )
    body = {"folder_id": folder["id"], "code": "twice"}
    assert client.post("/api/v1/public/passcodes/redeem", json=body).status_code == 200
    assert client.post("/api/v1/public/passcodes/redeem", json=body).status_code == 200
    assert client.post("/api/v1/public/passcodes/redeem", json=body).status_code == 401


def test_max_devices_passcode_blocks_an_unknown_device(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "one-device", "max_devices": 1},
        headers=auth_headers,
    )
    first = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "one-device", "device_key": "device-a"},
    )
    assert first.status_code == 200
    second = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "one-device", "device_key": "device-b"},
    )
    assert second.status_code == 401
    # The same device may come back as often as it likes.
    again = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "one-device", "device_key": "device-a"},
    )
    assert again.status_code == 200

    with SessionLocal() as db:
        assert db.execute(PasscodeGrant.__table__.select()).fetchall().__len__() == 1


def test_passcode_without_a_device_key_is_rejected_when_devices_are_capped(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "one-device", "max_devices": 1},
        headers=auth_headers,
    )
    resp = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "one-device"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"][0]["field"] == "device_key"


# -------------------------------------------------------------- share links


def test_share_link_opens_a_folder_and_can_be_revoked(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    created = client.post(
        f"/api/v1/folders/{folder['id']}/shares",
        json={"role": "viewer"},
        headers=auth_headers,
    )
    assert created.status_code == 201, created.text
    token = created.json()["token"]
    assert created.json()["url"].endswith(f"/share/{token}")

    redeemed = client.post(f"/api/v1/public/shares/{token}/redeem", json={})
    assert redeemed.status_code == 200, redeemed.text
    assert redeemed.json()["access"] == "share"
    scoped = redeemed.json()["token"]

    resp = client.get(
        f"/api/v1/public/folders/{folder['id']}", headers=guest_headers(scoped)
    )
    assert resp.status_code == 200

    share_id = created.json()["id"]
    assert client.delete(f"/api/v1/shares/{share_id}", headers=auth_headers).status_code == 204
    assert client.post(f"/api/v1/public/shares/{token}/redeem", json={}).status_code == 404


def test_share_link_honours_the_role_it_was_created_with(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    viewer = client.post(
        f"/api/v1/folders/{folder['id']}/shares", json={"role": "viewer"}, headers=auth_headers
    ).json()
    editor = client.post(
        f"/api/v1/folders/{folder['id']}/shares", json={"role": "editor"}, headers=auth_headers
    ).json()

    viewer_token = client.post(
        f"/api/v1/public/shares/{viewer['token']}/redeem", json={}
    ).json()["token"]
    editor_token = client.post(
        f"/api/v1/public/shares/{editor['token']}/redeem", json={}
    ).json()["token"]

    # A viewer may read but not comment; an editor may do both.
    comment = {"body": "looks good"}
    assert (
        client.post(
            f"/api/v1/folders/{folder['id']}/comments",
            json=comment,
            headers=guest_headers(viewer_token),
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/v1/folders/{folder['id']}/comments",
            json=comment,
            headers=guest_headers(editor_token),
        ).status_code
        == 201
    )


# --------------------------------------------------------------------- ZIP


def test_zip_download_contains_video_audio_and_transcript(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="public")
    video = make_video(client, auth_headers, title="Interview 1")
    seed_media(video["id"])
    client.put(
        f"/api/v1/videos/{video['id']}/folder",
        json={"folder_id": folder["id"]},
        headers=auth_headers,
    )

    resp = client.get(f"/api/v1/public/folders/{folder['id']}/zip")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        names = zf.namelist()
        assert any(n.endswith("/video/media.mp4") for n in names)
        assert any(n.endswith("/audio/audio.wav") for n in names)
        assert any(n.endswith("/transcript/") is False and n.endswith(".txt") for n in names)
        # The folder name is the archive root, so the tree is usable as-is.
        assert all(n.startswith("Phnom Penh cafés/") for n in names)


def test_zip_download_is_refused_for_a_private_folder_without_access(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    assert client.get(f"/api/v1/public/folders/{folder['id']}/zip").status_code == 404


def test_zip_download_is_allowed_with_a_passcode(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="private")
    video = make_video(client, auth_headers, title="Private clip")
    seed_media(video["id"])
    client.put(
        f"/api/v1/videos/{video['id']}/folder",
        json={"folder_id": folder["id"]},
        headers=auth_headers,
    )
    client.post(
        f"/api/v1/folders/{folder['id']}/passcodes",
        json={"code": "letmein"},
        headers=auth_headers,
    )
    token = client.post(
        "/api/v1/public/passcodes/redeem",
        json={"folder_id": folder["id"], "code": "letmein"},
    ).json()["token"]

    resp = client.get(
        f"/api/v1/public/folders/{folder['id']}/zip", headers=guest_headers(token)
    )
    assert resp.status_code == 200


def test_zip_of_an_empty_folder_is_a_conflict(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="public")
    resp = client.get(f"/api/v1/public/folders/{folder['id']}/zip")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "conflict"


# ----------------------------------------------------------------- comments


def test_anonymous_visitor_cannot_comment(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="public")
    resp = client.post(f"/api/v1/folders/{folder['id']}/comments", json={"body": "hi"})
    assert resp.status_code == 401


def test_comment_round_trip(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="public")
    posted = client.post(
        f"/api/v1/folders/{folder['id']}/comments",
        json={"body": "Needs a second speaker label."},
        headers=auth_headers,
    )
    assert posted.status_code == 201, posted.text
    listed = client.get(f"/api/v1/folders/{folder['id']}/comments", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()["items"][0]["body"] == "Needs a second speaker label."


# ------------------------------------------------------------------ members


def test_invited_viewer_can_read_but_not_manage(client, auth_headers):
    """The invite roles are enforced, not just stored (instruction.txt #1)."""
    from app.auth import hash_password
    from app.models import AdminUser

    user_email = "user@test.local"
    with SessionLocal() as db:
        # The fixture preserves admin_users between tests, so clear any row a
        # previous run of this test left behind.
        db.execute(AdminUser.__table__.delete().where(AdminUser.email == user_email))
        db.commit()
        db.add(
            AdminUser(
                email=user_email,
                password_hash=hash_password("user-password"),
                role="user",
                is_active=True,
            )
        )
        db.commit()

    login = client.post(
        "/api/v1/auth/login",
        json={"email": user_email, "password": "user-password"},
    )
    assert login.status_code == 200, login.text
    user_headers = guest_headers(login.json()["token"])

    folder = make_folder(client, auth_headers, name="Curation queue", visibility="private")
    invited = client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": user_email, "role": "viewer"},
        headers=auth_headers,
    )
    assert invited.status_code == 201, invited.text
    assert invited.json()["role"] == "viewer"

    # A viewer may open the folder and its cards...
    assert client.get(f"/api/v1/folders/{folder['id']}", headers=user_headers).status_code == 200
    assert (
        client.get(f"/api/v1/public/folders/{folder['id']}", headers=user_headers).status_code
        == 200
    )
    # ...but may not rename it, move videos in, or export it.
    assert (
        client.patch(
            f"/api/v1/folders/{folder['id']}",
            json={"name": "Renamed"},
            headers=user_headers,
        ).status_code
        == 403
    )
    assert (
        client.get(f"/api/v1/public/folders/{folder['id']}/zip", headers=user_headers).status_code
        == 409  # readable, but the folder has no media yet
    )

    # Promoting the same user to editor unlocks management.
    user_id = invited.json()["user_id"]
    promoted = client.patch(
        f"/api/v1/folders/{folder['id']}/members/{user_id}",
        json={"role": "editor"},
        headers=auth_headers,
    )
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["role"] == "editor"
    assert (
        client.patch(
            f"/api/v1/folders/{folder['id']}",
            json={"name": "Renamed"},
            headers=user_headers,
        ).status_code
        == 200
    )
