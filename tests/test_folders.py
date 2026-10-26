"""Folders: the unit a dataset is organised, published and exported.

Covers the CRUD surface, the nesting that the ZIP export mirrors, the
visibility rule (only the operator publishes), the member/comment roles, and
the dataset card listing that both the console and the guest portal read.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import Folder, Video
from tests.helpers import (
    create_folder,
    ensure_member,
    guest_headers,
    make_folder,
    make_transcript,
    member_headers,
    upload_video,
)


# ------------------------------------------------------------------ creating


def test_create_folder_returns_the_caller_as_editor(client, auth_headers):
    resp = create_folder(client, auth_headers, name="Cafés")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "Cafés"
    assert body["visibility"] == "private"
    assert body["my_role"] == "editor"
    assert body["owner_email"] == "admin@test.local"
    assert body["video_count"] == 0
    assert body["children"] == []


def test_create_folder_requires_a_signed_in_user(client):
    resp = create_folder(client, {}, name="Nope")
    assert resp.status_code == 401


def test_only_an_admin_may_publish_a_public_folder(client, auth_headers):
    """A public folder is readable by anonymous visitors, so only the operator
    may create one."""
    email = ensure_member()
    resp = create_folder(client, member_headers(email), name="Leak", visibility="public")
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"

    ok = create_folder(client, auth_headers, name="Open", visibility="public")
    assert ok.status_code == 201


def test_folder_name_is_required_and_length_bounded(client, auth_headers):
    assert create_folder(client, auth_headers, name="").status_code == 422
    assert create_folder(client, auth_headers, name="x" * 256).status_code == 422


# -------------------------------------------------------------------- nesting


def test_folders_nest_and_the_list_returns_a_tree(client, auth_headers):
    parent = make_folder(client, auth_headers, name="Parent")
    make_folder(client, auth_headers, name="Child", parent_id=parent["id"])

    resp = client.get("/api/v1/folders", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    roots = resp.json()["items"]
    assert len(roots) == 1
    assert roots[0]["name"] == "Parent"
    assert [c["name"] for c in roots[0]["children"]] == ["Child"]


def test_get_folder_includes_its_children(client, auth_headers):
    parent = make_folder(client, auth_headers, name="Parent")
    child = make_folder(client, auth_headers, name="Child", parent_id=parent["id"])

    resp = client.get(f"/api/v1/folders/{parent['id']}", headers=auth_headers)
    assert resp.status_code == 200
    assert [c["id"] for c in resp.json()["children"]] == [child["id"]]


def test_a_child_folder_inherits_the_parents_visibility(client, auth_headers):
    """A member of a private parent may browse its sub-folders (#6)."""
    parent = make_folder(client, auth_headers, name="Parent")
    child = make_folder(client, auth_headers, name="Child", parent_id=parent["id"])
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{parent['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    resp = client.get(f"/api/v1/folders/{child['id']}", headers=member_headers(email))
    assert resp.status_code == 200, resp.text
    assert resp.json()["my_role"] == "viewer"


def test_creating_a_child_requires_manage_on_the_parent(client, auth_headers):
    parent = make_folder(client, auth_headers, name="Parent")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{parent['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    resp = create_folder(
        client, member_headers(email), name="Sneaky", parent_id=parent["id"]
    )
    assert resp.status_code == 403


# ------------------------------------------------------------------ visibility


def test_a_stranger_cannot_see_a_private_folder(client, auth_headers):
    """Deliberately a 404, not a 403: whether the folder exists is private."""
    folder = make_folder(client, auth_headers, name="Secret")
    resp = client.get(f"/api/v1/folders/{folder['id']}", headers=guest_headers())
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_a_stranger_can_see_a_public_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Open", visibility="public")
    resp = client.get(f"/api/v1/folders/{folder['id']}", headers=guest_headers())
    assert resp.status_code == 200
    assert resp.json()["my_role"] == "viewer"


def test_listing_hides_private_folders_from_a_stranger(client, auth_headers):
    make_folder(client, auth_headers, name="Secret")
    make_folder(client, auth_headers, name="Open", visibility="public")

    resp = client.get("/api/v1/folders", headers=guest_headers())
    assert resp.status_code == 200
    assert [f["name"] for f in resp.json()["items"]] == ["Open"]


def test_listing_is_not_required_for_a_public_folder(client, auth_headers):
    """The guest portal's entry point works with no Authorization header."""
    make_folder(client, auth_headers, name="Open", visibility="public")
    resp = client.get("/api/v1/folders")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


# -------------------------------------------------------------------- updating


def test_rename_and_reposition_a_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Old")
    resp = client.patch(
        f"/api/v1/folders/{folder['id']}",
        json={"name": "New", "position": 5},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "New"
    assert resp.json()["position"] == 5


def test_a_folder_cannot_be_its_own_parent(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Vanity")
    resp = client.patch(
        f"/api/v1/folders/{folder['id']}",
        json={"parent_id": folder["id"]},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"][0]["issue"] == "self_reference"


def test_moving_a_folder_requires_manage_on_the_destination(client, auth_headers):
    source = make_folder(client, auth_headers, name="Source")
    dest = make_folder(client, auth_headers, name="Destination")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{source['id']}/members",
        json={"email": email, "role": "editor"},
        headers=auth_headers,
    )
    # The member may move *their* folder, but not into a folder they do not own.
    resp = client.patch(
        f"/api/v1/folders/{source['id']}",
        json={"parent_id": dest["id"]},
        headers=member_headers(email),
    )
    assert resp.status_code == 403


def test_only_an_admin_may_make_an_existing_folder_public(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Draft")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "editor"},
        headers=auth_headers,
    )
    resp = client.patch(
        f"/api/v1/folders/{folder['id']}",
        json={"visibility": "public"},
        headers=member_headers(email),
    )
    assert resp.status_code == 403


def test_update_rejects_an_unknown_folder(client, auth_headers):
    from uuid import uuid4

    resp = client.patch(f"/api/v1/folders/{uuid4()}", json={"name": "X"}, headers=auth_headers)
    assert resp.status_code == 404


# -------------------------------------------------------------------- deleting


def test_delete_folder_cascades_to_children_but_keeps_the_videos(client, auth_headers):
    """Removing a grouping must never destroy the underlying media."""
    parent = make_folder(client, auth_headers, name="Parent")
    child = make_folder(client, auth_headers, name="Child", parent_id=parent["id"])
    video = upload_video(client, auth_headers, folder_id=child["id"], title="Kept")

    resp = client.delete(f"/api/v1/folders/{parent['id']}", headers=auth_headers)
    assert resp.status_code == 204

    with SessionLocal() as db:
        assert db.get(Folder, parent["id"]) is None
        assert db.get(Folder, child["id"]) is None
        # The video survives, detached from the folder it was filed under.
        surviving = db.get(Video, video["id"])
        assert surviving is not None
        assert surviving.folder_id is None


def test_delete_requires_manage(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Keep out")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    resp = client.delete(f"/api/v1/folders/{folder['id']}", headers=member_headers(email))
    assert resp.status_code == 403


# --------------------------------------------------------------- filing videos


def test_place_a_video_into_a_folder_and_back_out(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    video = upload_video(client, auth_headers, title="Unfiled")

    placed = client.put(
        f"/api/v1/videos/{video['id']}/folder",
        json={"folder_id": folder["id"], "position": 3},
        headers=auth_headers,
    )
    assert placed.status_code == 200, placed.text
    assert placed.json()["folder_id"] == folder["id"]
    assert placed.json()["position"] == 3

    removed = client.put(
        f"/api/v1/videos/{video['id']}/folder", json={"folder_id": None}, headers=auth_headers
    )
    assert removed.status_code == 200
    assert removed.json()["folder_id"] is None


def test_filing_a_video_requires_manage_on_the_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    video = upload_video(client, auth_headers, title="Unfiled")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    resp = client.put(
        f"/api/v1/videos/{video['id']}/folder",
        json={"folder_id": folder["id"]},
        headers=member_headers(email),
    )
    assert resp.status_code == 403


def test_ingesting_directly_into_a_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    video = upload_video(client, auth_headers, folder_id=folder["id"], title="Filed")
    assert video["folder_id"] == folder["id"]
    assert video["position"] == 0


def test_ingesting_into_a_folder_the_caller_cannot_edit_is_refused(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    from tests.helpers import MEDIA_BYTES

    resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("clip.mp4", MEDIA_BYTES, "video/mp4")},
        data={"folder_id": folder["id"]},
        headers=member_headers(email),
    )
    assert resp.status_code == 403


def test_videos_are_listed_within_a_folder_in_position_order(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    second = upload_video(client, auth_headers, folder_id=folder["id"], title="Second")
    first = upload_video(client, auth_headers, folder_id=folder["id"], title="First")
    client.put(
        f"/api/v1/videos/{first['id']}/folder",
        json={"folder_id": folder["id"], "position": 0},
        headers=auth_headers,
    )
    client.put(
        f"/api/v1/videos/{second['id']}/folder",
        json={"folder_id": folder["id"], "position": 1},
        headers=auth_headers,
    )

    resp = client.get(f"/api/v1/folders/{folder['id']}/videos", headers=auth_headers)
    assert resp.status_code == 200
    assert [v["title"] for v in resp.json()] == ["First", "Second"]


def test_video_count_reflects_the_folder_contents(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    upload_video(client, auth_headers, folder_id=folder["id"], title="One")
    resp = client.get(f"/api/v1/folders/{folder['id']}", headers=auth_headers)
    assert resp.json()["video_count"] == 1


# ------------------------------------------------------------- dataset cards


def test_dataset_cards_include_the_raw_cleaned_badge(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    raw = upload_video(client, auth_headers, folder_id=folder["id"], title="Raw")
    cleaned = upload_video(client, auth_headers, folder_id=folder["id"], title="Cleaned")
    make_transcript(client, auth_headers, raw["id"], "សួស្តី", cleaned=False)
    make_transcript(client, auth_headers, cleaned["id"], "corrected", cleaned=True)

    resp = client.get("/api/v1/dataset/cards", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    badges = {c["title"]: c["badge"] for c in resp.json()["items"]}
    assert badges == {"Raw": "raw", "Cleaned": "cleaned"}


def test_dataset_cards_can_be_filtered_to_cleaned_only(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    raw = upload_video(client, auth_headers, folder_id=folder["id"], title="Raw")
    cleaned = upload_video(client, auth_headers, folder_id=folder["id"], title="Cleaned")
    make_transcript(client, auth_headers, raw["id"], "raw text", cleaned=False)
    make_transcript(client, auth_headers, cleaned["id"], "fixed text", cleaned=True)

    resp = client.get("/api/v1/dataset/cards?cleaned=true", headers=auth_headers)
    items = resp.json()["items"]
    assert [c["title"] for c in items] == ["Cleaned"]


def test_dataset_cards_search_matches_title_and_source(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    upload_video(client, auth_headers, folder_id=folder["id"], title="Phnom Penh café")
    upload_video(client, auth_headers, folder_id=folder["id"], title="Siem Reap")

    resp = client.get("/api/v1/dataset/cards?q=phnom", headers=auth_headers)
    assert [c["title"] for c in resp.json()["items"]] == ["Phnom Penh café"]


def test_dataset_cards_narrow_to_one_folder(client, auth_headers):
    a = make_folder(client, auth_headers, name="A")
    b = make_folder(client, auth_headers, name="B")
    upload_video(client, auth_headers, folder_id=a["id"], title="In A")
    upload_video(client, auth_headers, folder_id=b["id"], title="In B")

    resp = client.get(f"/api/v1/dataset/cards?folder_id={a['id']}", headers=auth_headers)
    assert [c["title"] for c in resp.json()["items"]] == ["In A"]


def test_a_stranger_only_sees_cards_from_public_folders(client, auth_headers):
    """The guest portal must not leak a private folder's videos."""
    public = make_folder(client, auth_headers, name="Open", visibility="public")
    private = make_folder(client, auth_headers, name="Secret")
    upload_video(client, auth_headers, folder_id=public["id"], title="Visible")
    upload_video(client, auth_headers, folder_id=private["id"], title="Hidden")

    resp = client.get("/api/v1/dataset/cards", headers=guest_headers())
    assert resp.status_code == 200
    assert [c["title"] for c in resp.json()["items"]] == ["Visible"]


def test_a_member_sees_their_own_unfiled_videos(client, auth_headers):
    """The ingesting account keeps access to what it filed, even unfiled."""
    email = ensure_member()
    upload_video(client, member_headers(email), title="Mine")
    resp = client.get("/api/v1/dataset/cards", headers=member_headers(email))
    assert [c["title"] for c in resp.json()["items"]] == ["Mine"]


def test_dataset_cards_are_paginated(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    for i in range(5):
        upload_video(client, auth_headers, folder_id=folder["id"], title=f"clip {i}")

    resp = client.get("/api/v1/dataset/cards?page=2&page_size=2", headers=auth_headers)
    body = resp.json()
    assert body["total"] == 5
    assert body["page"] == 2
    assert len(body["items"]) == 2


def test_dataset_cards_reject_an_invalid_page_size(client, auth_headers):
    resp = client.get("/api/v1/dataset/cards?page_size=1000", headers=auth_headers)
    assert resp.status_code == 422


# ------------------------------------------------------------------- members


def test_invite_creates_a_placeholder_account_for_an_unknown_email(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    resp = client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": "newcomer@test.local", "role": "viewer"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["email"] == "newcomer@test.local"
    assert resp.json()["role"] == "viewer"

    # The invitation is waiting when they first sign in: the placeholder row is
    # created by the invite itself, before the person has any password.
    listed = client.get(f"/api/v1/folders/{folder['id']}/members", headers=auth_headers)
    assert [m["email"] for m in listed.json()["items"]] == ["newcomer@test.local"]


def test_invite_rejects_a_malformed_email(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    resp = client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": "not-an-email"},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"][0]["field"] == "email"


def test_invite_requires_manage_on_the_folder(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "viewer"},
        headers=auth_headers,
    )
    resp = client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": "another@test.local"},
        headers=member_headers(email),
    )
    assert resp.status_code == 403


def test_reinviting_updates_the_role_instead_of_duplicating(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": "person@test.local", "role": "viewer"},
        headers=auth_headers,
    )
    resp = client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": "person@test.local", "role": "editor"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    listed = client.get(f"/api/v1/folders/{folder['id']}/members", headers=auth_headers)
    assert len(listed.json()["items"]) == 1
    assert listed.json()["items"][0]["role"] == "editor"


def test_update_and_remove_a_member(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": "person@test.local", "role": "viewer"},
        headers=auth_headers,
    )
    with SessionLocal() as db:
        from app.models import AdminUser, FolderMember

        user_id = db.execute(
            select(AdminUser).where(AdminUser.email == "person@test.local")
        ).scalar_one().id

    updated = client.patch(
        f"/api/v1/folders/{folder['id']}/members/{user_id}",
        json={"role": "commenter"},
        headers=auth_headers,
    )
    assert updated.status_code == 200
    assert updated.json()["role"] == "commenter"

    removed = client.delete(
        f"/api/v1/folders/{folder['id']}/members/{user_id}", headers=auth_headers
    )
    assert removed.status_code == 204
    listed = client.get(f"/api/v1/folders/{folder['id']}/members", headers=auth_headers)
    assert listed.json()["items"] == []


def test_updating_an_unknown_member_is_a_404(client, auth_headers):
    from uuid import uuid4

    folder = make_folder(client, auth_headers, name="Dataset")
    resp = client.patch(
        f"/api/v1/folders/{folder['id']}/members/{uuid4()}",
        json={"role": "editor"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_a_viewer_cannot_read_the_member_list_of_a_folder_they_are_not_in(
    client, auth_headers
):
    folder = make_folder(client, auth_headers, name="Dataset")
    email = ensure_member()
    resp = client.get(
        f"/api/v1/folders/{folder['id']}/members", headers=member_headers(email)
    )
    assert resp.status_code == 404


# ------------------------------------------------------------------ comments


def test_a_commenter_may_comment_but_a_viewer_may_not(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    commenter = ensure_member("commenter")
    viewer = ensure_member("viewer")
    for email, role in ((commenter, "commenter"), (viewer, "viewer")):
        client.post(
            f"/api/v1/folders/{folder['id']}/members",
            json={"email": email, "role": role},
            headers=auth_headers,
        )

    denied = client.post(
        f"/api/v1/folders/{folder['id']}/comments",
        json={"body": "nice"},
        headers=member_headers(viewer),
    )
    assert denied.status_code == 403
    assert denied.json()["error"]["details"][0]["role"] == "viewer"

    allowed = client.post(
        f"/api/v1/folders/{folder['id']}/comments",
        json={"body": "nice"},
        headers=member_headers(commenter),
    )
    assert allowed.status_code == 201, allowed.text
    assert allowed.json()["body"] == "nice"


def test_comments_are_listed_oldest_first(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset", visibility="public")
    email = ensure_member()
    client.post(
        f"/api/v1/folders/{folder['id']}/members",
        json={"email": email, "role": "commenter"},
        headers=auth_headers,
    )
    headers = member_headers(email)
    client.post(f"/api/v1/folders/{folder['id']}/comments", json={"body": "first"}, headers=headers)
    client.post(f"/api/v1/folders/{folder['id']}/comments", json={"body": "second"}, headers=headers)

    resp = client.get(f"/api/v1/folders/{folder['id']}/comments", headers=headers)
    assert [c["body"] for c in resp.json()["items"]] == ["first", "second"]


def test_a_public_folder_is_commentable_by_a_member_of_the_public(client, auth_headers):
    """Public means readable; writing still needs the commenter role."""
    folder = make_folder(client, auth_headers, name="Open", visibility="public")
    resp = client.post(
        f"/api/v1/folders/{folder['id']}/comments",
        json={"body": "hello"},
        headers=guest_headers(),
    )
    assert resp.status_code == 403


def test_an_empty_comment_is_rejected(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    resp = client.post(
        f"/api/v1/folders/{folder['id']}/comments", json={"body": ""}, headers=auth_headers
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------- export


def test_folder_transcript_metadata_lists_every_version(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    video = upload_video(client, auth_headers, folder_id=folder["id"], title="Unit")
    make_transcript(client, auth_headers, video["id"], "v1", cleaned=False)
    make_transcript(client, auth_headers, video["id"], "v2", cleaned=True)

    resp = client.get(f"/api/v1/folders/{folder['id']}/transcripts", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    row = resp.json()["videos"][0]
    assert row["video_id"] == video["id"]
    # newest first
    assert [t["version"] for t in row["transcripts"]] == [2, 1]
    assert row["transcripts"][0]["cleaned"] is True


def test_download_zip_returns_a_real_archive(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Dataset")
    video = upload_video(client, auth_headers, folder_id=folder["id"], title="Unit")
    make_transcript(client, auth_headers, video["id"], "the transcript")

    resp = client.get(f"/api/v1/folders/{folder['id']}/download", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"
    assert "no-store" in resp.headers["cache-control"]

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        assert any(n.endswith("/video/media.mp4") for n in names)
        assert any(n.endswith(".txt") for n in names)
        # Audio is included only once extraction has produced it; the upload
        # path queues that job but these tests never run the worker.
        assert not any(n.endswith("/audio/audio.wav") for n in names)


def test_download_zip_of_a_folder_with_nothing_to_export_is_a_409(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Empty")
    resp = client.get(f"/api/v1/folders/{folder['id']}/download", headers=auth_headers)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "nothing_to_export"
    assert resp.json()["error"]["details"][0]["folder_id"] == folder["id"]


def test_download_zip_is_refused_for_a_folder_the_caller_cannot_read(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Secret")
    resp = client.get(f"/api/v1/folders/{folder['id']}/download", headers=guest_headers())
    assert resp.status_code == 404
