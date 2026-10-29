"""F2 black-box: the developer acts on media it can already see (contract §0).

The confirmed decision is **library-scoped, not global**: the library tier's
extra media right applies only to units the caller can already read — public
folders, plus private folders where they hold a role. An ungranted private
folder stays invisible (404 on read, absent from listings), so there is
nothing to delete, move, retry or ingest into. The same widening is
deliberately narrow: folder administration and transcript editing are NOT
granted to the developer, and the reviewer is a no-op (delete still 403,
transcript edit still works).

These are black-box tests through the real routes, mirroring
`tests/test_authorization.py`; roles are database-authoritative, so each actor
logs in normally.
"""

from __future__ import annotations

from uuid import uuid4

from app.auth import hash_password
from app.db import SessionLocal
from app.models import AdminUser
from tests.helpers import ensure_user, make_folder, media_bytes, upload_video, user_headers

API = "/api/v1"
PASSWORD = "media-pass-12345"


def _headers_for(client, role: str) -> tuple[dict, str]:
    """A signed-in account with the given role, plus its email.

    Roles are database-authoritative, so logging in normally is the honest way
    to mint the token: whatever the JWT claims, the row decides.
    """
    email = f"{role}-{uuid4().hex[:8]}@test.local"
    with SessionLocal() as db:
        db.add(
            AdminUser(
                email=email,
                role=role,
                password_hash=hash_password(PASSWORD),
                is_active=True,
            )
        )
        db.commit()
    resp = client.post(
        f"{API}/auth/login", json={"email": email, "password": PASSWORD}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}, email


def _invite(client, admin, folder_id, email: str, role: str = "editor"):
    resp = client.post(
        f"{API}/folders/{folder_id}/members",
        json={"email": email, "role": role},
        headers=admin,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ----------------------------------------------------------- delete / retry


def test_developer_deletes_a_video_in_a_public_folder_even_without_a_grant(client, auth_headers):
    dev, _ = _headers_for(client, "developer")
    folder = make_folder(client, auth_headers, name="Open", visibility="public")
    video = upload_video(client, auth_headers, folder_id=folder["id"])

    resp = client.delete(f"{API}/videos/{video['id']}", headers=dev)
    assert resp.status_code == 204, resp.text
    assert client.get(f"{API}/videos/{video['id']}", headers=auth_headers).status_code == 404


def test_developer_deletes_a_video_in_a_private_folder_where_they_are_a_member(client, auth_headers):
    dev, dev_email = _headers_for(client, "developer")
    folder = make_folder(client, auth_headers, name="Vault", visibility="private")
    video = upload_video(client, auth_headers, folder_id=folder["id"])
    _invite(client, auth_headers, folder["id"], dev_email, role="editor")

    resp = client.delete(f"{API}/videos/{video['id']}", headers=dev)
    assert resp.status_code == 204, resp.text


def test_developer_cannot_delete_a_video_in_an_ungranted_private_folder(client, auth_headers):
    dev, _ = _headers_for(client, "developer")
    folder = make_folder(client, auth_headers, name="Vault", visibility="private")
    video = upload_video(client, auth_headers, folder_id=folder["id"])

    # The unit is invisible: the guard answers 403 and the folder itself 404.
    resp = client.delete(f"{API}/videos/{video['id']}", headers=dev)
    assert resp.status_code == 403, resp.text
    assert client.get(f"{API}/videos/{video['id']}", headers=dev).status_code == 403
    assert client.get(f"{API}/folders/{folder['id']}", headers=dev).status_code == 404
    assert client.get(f"{API}/videos/{video['id']}", headers=auth_headers).status_code == 200


def test_developer_moves_and_retries_a_visible_video(client, auth_headers):
    dev, _ = _headers_for(client, "developer")
    source = make_folder(client, auth_headers, name="Source", visibility="public")
    dest = make_folder(client, auth_headers, name="Dest", visibility="public")
    video = upload_video(client, auth_headers, folder_id=source["id"])

    # PATCH move, then PUT /videos/{id}/folder, both endpoints swapped.
    patched = client.patch(
        f"{API}/videos/{video['id']}", json={"folder_id": dest["id"]}, headers=dev
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["folder_id"] == dest["id"]

    placed = client.put(
        f"{API}/videos/{video['id']}/folder",
        json={"folder_id": source["id"]},
        headers=dev,
    )
    assert placed.status_code == 200, placed.text
    assert placed.json()["folder_id"] == source["id"]

    retried = client.post(f"{API}/videos/{video['id']}/retry", headers=dev)
    assert retried.status_code == 202, retried.text


# ------------------------------------------- no over-grant / no transcript


def test_developer_cannot_administer_a_visible_folder(client, auth_headers):
    """Every `require_manage` route stays closed: rename, delete, members,
    share, passcode. This is the over-grant the media tier must not cause."""
    dev, _ = _headers_for(client, "developer")
    folder = make_folder(client, auth_headers, name="Open", visibility="public")
    fid = folder["id"]

    assert client.patch(
        f"{API}/folders/{fid}", json={"name": "Renamed"}, headers=dev
    ).status_code == 403
    assert client.delete(f"{API}/folders/{fid}", headers=dev).status_code == 403
    assert client.post(
        f"{API}/folders/{fid}/members",
        json={"email": f"invitee-{uuid4().hex[:6]}@test.local", "role": "viewer"},
        headers=dev,
    ).status_code == 403
    assert client.post(
        f"{API}/folders/{fid}/shares", json={"role": "viewer"}, headers=dev
    ).status_code == 403
    assert client.post(
        f"{API}/folders/{fid}/passcodes", json={"code": "1234"}, headers=dev
    ).status_code == 403

    # The folder is untouched and still exists for its owner.
    assert client.get(f"{API}/folders/{fid}", headers=auth_headers).status_code == 200


def test_developer_cannot_edit_a_transcript_on_a_visible_unit(client, auth_headers):
    """The transcript capability is separate (`TRANSCRIPT_EDIT_ROLES`); the
    developer must not inherit it from the media widening."""
    dev, _ = _headers_for(client, "developer")
    folder = make_folder(client, auth_headers, name="Open", visibility="public")
    video = upload_video(client, auth_headers, folder_id=folder["id"])

    put = client.put(
        f"{API}/videos/{video['id']}/transcript",
        json={"content": "developer was here"},
        headers=dev,
    )
    assert put.status_code == 403, put.text
    patch = client.patch(
        f"{API}/videos/{video['id']}/transcript", json={"cleaned": True}, headers=dev
    )
    assert patch.status_code == 403, patch.text


# ------------------------------------------------------------- ingest / fork


def test_developer_can_ingest_and_copy_into_a_visible_folder(client, auth_headers):
    dev, _ = _headers_for(client, "developer")
    public = make_folder(client, auth_headers, name="Open", visibility="public")
    other = make_folder(client, auth_headers, name="Other", visibility="public")

    # Link ingest into a public folder: the destination is visible.
    resp = client.post(
        f"{API}/videos",
        json={
            "source_url": f"https://example.com/dev-{uuid4().hex}.mp4",
            "folder_id": public["id"],
        },
        headers=dev,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["folder_id"] == public["id"]

    # File upload into a public folder.
    uploaded = upload_video(client, dev, folder_id=public["id"], title="Dev upload")
    assert uploaded["folder_id"] == public["id"]

    # Fork a readable source into another visible destination.
    source = upload_video(client, auth_headers, folder_id=public["id"], title="Source")
    copied = client.post(
        f"{API}/videos/{source['id']}/copy",
        json={"folder_id": other["id"]},
        headers=dev,
    )
    assert copied.status_code == 201, copied.text
    assert copied.json()["folder_id"] == other["id"]


def test_developer_cannot_ingest_into_an_invisible_private_folder(client, auth_headers):
    dev, _ = _headers_for(client, "developer")
    public = make_folder(client, auth_headers, name="Open", visibility="public")
    private = make_folder(client, auth_headers, name="Vault", visibility="private")

    link = client.post(
        f"{API}/videos",
        json={
            "source_url": f"https://example.com/dev-{uuid4().hex}.mp4",
            "folder_id": private["id"],
        },
        headers=dev,
    )
    assert link.status_code == 403, link.text

    upload = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.mp4", media_bytes(), "video/mp4")},
        data={"folder_id": str(private["id"]), "start_transcription": "false"},
        headers=dev,
    )
    assert upload.status_code == 403, upload.text

    source = upload_video(client, auth_headers, folder_id=public["id"], title="Source")
    copied = client.post(
        f"{API}/videos/{source['id']}/copy",
        json={"folder_id": private["id"]},
        headers=dev,
    )
    assert copied.status_code == 403, copied.text


# ---------------------------------------------------- reviewer / plain user


def test_reviewer_delete_stays_forbidden_but_transcript_edit_still_works(client, auth_headers):
    reviewer, _ = _headers_for(client, "reviewer")
    folder = make_folder(client, auth_headers, name="Open", visibility="public")
    video = upload_video(client, auth_headers, folder_id=folder["id"])

    deleted = client.delete(f"{API}/videos/{video['id']}", headers=reviewer)
    assert deleted.status_code == 403, deleted.text

    edited = client.put(
        f"{API}/videos/{video['id']}/transcript",
        json={"content": "reviewed transcript"},
        headers=reviewer,
    )
    assert edited.status_code == 201, edited.text


def test_plain_user_deletes_their_own_video_but_not_someone_elses(client, auth_headers):
    headers = user_headers(ensure_user("media-owner"))

    own = upload_video(client, headers, title="Mine")
    assert client.delete(f"{API}/videos/{own['id']}", headers=headers).status_code == 204

    folder = make_folder(client, auth_headers, name="Open", visibility="public")
    someone_elses = upload_video(client, auth_headers, folder_id=folder["id"])
    denied = client.delete(f"{API}/videos/{someone_elses['id']}", headers=headers)
    assert denied.status_code == 403, denied.text
