"""Authorization regressions: routes that authenticated but never authorized.

Each test here pins a hole that was found by reading the routers, where the
guard was `get_actor` (a valid token is enough) instead of a role check. The
important property is the *contrast*: the route that was already correct keeps
working, and the one that was wide now agrees with it.

The member used throughout holds a real account row but no folder membership,
no share token and no passcode grant — the weakest signed-in identity.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AdminUser, Folder
from tests.conftest import ADMIN_EMAIL
from tests.helpers import ensure_member, member_headers

API = "/api/v1"


def _admin(client):
    from tests.conftest import ADMIN_PASSWORD

    resp = client.post(
        f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _private_folder(client, admin, *, name=None):
    resp = client.post(
        f"{API}/folders",
        json={"name": name or f"Private {uuid4().hex[:6]}", "visibility": "private"},
        headers=admin,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _video_in(client, admin, folder_id, *, title="Clip"):
    from tests.helpers import media_bytes

    resp = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.mp4", media_bytes(), "video/mp4")},
        data={"title": title, "folder_id": str(folder_id),
              "start_transcription": "false"},
        headers=admin,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _unfiled_video(client, admin):
    from tests.helpers import media_bytes

    resp = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.mp4", media_bytes(), "video/mp4")},
        data={"title": f"Unfiled {uuid4().hex[:6]}", "start_transcription": "false"},
        headers=admin,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# ----------------------------------------------------------- video moves


def test_a_member_cannot_detach_a_video_from_a_folder_they_cannot_see(client):
    """The hole: `folder_id: null` was applied with no check on the source."""
    admin = _admin(client)
    folder = _private_folder(client, admin)
    video = _video_in(client, admin, folder["id"])
    headers = member_headers(ensure_member("detacher"))

    # The member cannot even read the folder, so the video is invisible to them.
    assert client.get(f"{API}/folders/{folder['id']}", headers=headers).status_code == 404

    resp = client.put(
        f"{API}/videos/{video['id']}/folder", json={"folder_id": None}, headers=headers
    )
    assert resp.status_code == 403, resp.text

    # The video is still filed where the admin put it.
    with SessionLocal() as db:
        assert db.get(__import__("app.models", fromlist=["Video"]).Video,
                      video["id"]).folder_id is not None


def test_a_member_cannot_move_a_video_into_a_folder_they_do_not_manage(client):
    admin = _admin(client)
    folder = _private_folder(client, admin)
    video = _video_in(client, admin, folder["id"])
    headers = member_headers(ensure_member("mover"))

    resp = client.put(
        f"{API}/videos/{video['id']}/folder",
        json={"folder_id": str(folder["id"])},
        headers=headers,
    )
    assert resp.status_code == 403, resp.text


def test_an_editor_of_another_folder_cannot_steal_a_video(client):
    """Cross-folder write: managing folder B must not grant access to folder A."""
    admin = _admin(client)
    source = _private_folder(client, admin, name=f"Source {uuid4().hex[:6]}")
    target = _private_folder(client, admin, name=f"Target {uuid4().hex[:6]}")
    video = _video_in(client, admin, source["id"])

    editor_email = ensure_member("cross-editor")
    with SessionLocal() as db:
        from app.models import FolderMember

        db.add(FolderMember(folder_id=target["id"], user_id=db.get(
            AdminUser, db.execute(select(AdminUser).where(
                AdminUser.email == editor_email)).scalar_one().id).id,
            role="editor"))
        db.commit()

    headers = member_headers(editor_email)
    resp = client.put(
        f"{API}/videos/{video['id']}/folder",
        json={"folder_id": str(target["id"])},
        headers=headers,
    )
    assert resp.status_code == 403, resp.text


def test_patch_video_rejects_a_destination_the_caller_cannot_manage(client):
    """`PATCH /videos/{id}` accepts `folder_id`; it must check the destination."""
    admin = _admin(client)
    source = _private_folder(client, admin, name=f"Src {uuid4().hex[:6]}")
    target = _private_folder(client, admin, name=f"Dst {uuid4().hex[:6]}")
    video = _video_in(client, admin, source["id"])
    headers = member_headers(ensure_member("patcher"))

    resp = client.patch(
        f"{API}/videos/{video['id']}",
        json={"folder_id": str(target["id"])},
        headers=headers,
    )
    assert resp.status_code == 403, resp.text

    # The member cannot edit a video in a folder they cannot read at all.
    resp = client.patch(
        f"{API}/videos/{video['id']}", json={"notes": "hello"}, headers=headers
    )
    assert resp.status_code == 403, resp.text


# ----------------------------------------------------------- unfiled media


def test_unfiled_media_is_not_served_to_a_member_who_is_not_the_creator(client):
    """Metadata 403s but media/audio used to return the bytes."""
    admin = _admin(client)
    video = _unfiled_video(client, admin)
    headers = member_headers(ensure_member("eavesdropper"))

    assert client.get(f"{API}/videos/{video['id']}", headers=headers).status_code == 403
    assert client.get(f"{API}/videos/{video['id']}/media", headers=headers).status_code == 404
    assert client.get(f"{API}/videos/{video['id']}/audio", headers=headers).status_code == 404


def test_unfiled_media_is_served_to_its_creator(client):
    """The creator rule is preserved, not just narrowed."""
    admin = _admin(client)
    video = _unfiled_video(client, admin)
    assert client.get(f"{API}/videos/{video['id']}/media", headers=admin).status_code == 200


# ----------------------------------------------------------- operator data


def test_jobs_and_stats_are_operator_only(client):
    """A member must not read the queue that serves every folder."""
    admin = _admin(client)
    _private_folder(client, admin)
    headers = member_headers(ensure_member("queue-peeker"))

    for path in (f"{API}/jobs", f"{API}/stats"):
        resp = client.get(path, headers=headers)
        assert resp.status_code == 403, f"{path}: {resp.text}"
        assert client.get(path, headers=admin).status_code == 200, path


def test_a_member_cannot_mutate_the_operator_queue(client):
    admin = _admin(client)
    folder = _private_folder(client, admin)
    video = _video_in(client, admin, folder["id"])
    from tests.helpers import media_bytes  # noqa: F401  (video already ingested)

    # Produce a job row to target.
    client.post(
        f"{API}/videos/{video['id']}/transcript/generate",
        json={"engine": "whisper"},
        headers=admin,
    )
    jobs = client.get(f"{API}/jobs", headers=admin).json()["items"]
    assert jobs, "expected at least one job"

    headers = member_headers(ensure_member("job-tampering"))
    resp = client.post(f"{API}/jobs/{jobs[0]['id']}/cancel", headers=headers)
    assert resp.status_code == 403, resp.text


def test_merge_routes_require_the_video_role(client):
    """Merges are a video sub-resource and must inherit its role."""
    admin = _admin(client)
    folder = _private_folder(client, admin)
    video = _video_in(client, admin, folder["id"])
    headers = member_headers(ensure_member("merge-peeker"))

    resp = client.get(f"{API}/videos/{video['id']}/merges", headers=headers)
    assert resp.status_code == 403, resp.text

    resp = client.post(
        f"{API}/videos/{video['id']}/transcript/merge", json={}, headers=headers
    )
    assert resp.status_code == 403, resp.text


# ----------------------------------------------------------- public listing


def test_public_folder_total_counts_roots_not_the_forest(client):
    """`items` holds roots, so `total` must too."""
    admin = _admin(client)
    root = client.post(
        f"{API}/folders",
        json={"name": f"Root {uuid4().hex[:6]}", "visibility": "public"},
        headers=admin,
    ).json()
    client.post(
        f"{API}/folders",
        json={
            "name": f"Child {uuid4().hex[:6]}",
            "visibility": "public",
            "parent_id": str(root["id"]),
        },
        headers=admin,
    )

    data = client.get(f"{API}/public/folders").json()
    assert data["total"] == len(data["items"]), data
    assert data["total"] == 1, data


# ----------------------------------------------------------- folder cycles


def test_a_folder_cannot_be_moved_into_its_own_sub_folder(client):
    admin = _admin(client)
    root = client.post(
        f"{API}/folders",
        json={"name": f"A {uuid4().hex[:6]}", "visibility": "private"},
        headers=admin,
    ).json()
    child = client.post(
        f"{API}/folders",
        json={
            "name": f"B {uuid4().hex[:6]}",
            "visibility": "private",
            "parent_id": str(root["id"]),
        },
        headers=admin,
    ).json()

    resp = client.patch(
        f"{API}/folders/{root['id']}",
        json={"parent_id": str(child["id"])},
        headers=admin,
    )
    assert resp.status_code == 422, resp.text

    # The tree is still intact: the root is a root.
    assert client.get(f"{API}/folders", headers=admin).json()["total"] == 1


def test_cleaned_false_keeps_units_with_no_transcript(client):
    """`_card` badges a transcript-less unit "raw", so the filter must agree."""
    admin = _admin(client)
    folder = _private_folder(client, admin, name=f"Raw {uuid4().hex[:6]}")
    _video_in(client, admin, folder["id"], title=f"No transcript {uuid4().hex[:6]}")

    everything = client.get(
        f"{API}/dataset/cards", params={"folder_id": str(folder["id"])}, headers=admin
    ).json()
    assert everything["total"] == 1
    assert everything["items"][0]["badge"] == "raw"

    raw = client.get(
        f"{API}/dataset/cards",
        params={"folder_id": str(folder["id"]), "cleaned": "false"},
        headers=admin,
    ).json()
    assert raw["total"] == 1, raw


def test_the_created_by_index_exists(client):
    """A hot filter column; `create_all` alone would not add it to an old DB."""
    from sqlalchemy import text

    from app.db import engine

    with engine.connect() as conn:
        names = {
            row[0]
            for row in conn.execute(
                text("select indexname from pg_indexes where tablename = 'videos'")
            ).all()
        }
    assert "ix_videos_created_by" in names, names
