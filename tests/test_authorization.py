"""Authorization regressions: routes that authenticated but never authorized.

Each test here pins a hole that was found by reading the routers, where the
guard was `get_actor` (a valid token is enough) instead of a role check. The
important property is the *contrast*: the route that was already correct keeps
working, and the one that was wide now agrees with it.

The user used throughout holds a real account row but no folder membership,
no share token and no passcode grant — the weakest signed-in identity. The
role-tier tests at the bottom pin the five-role system (`app.roles`): who may
operate the pipeline, who may read the whole library, and who may correct a
transcript on a video they cannot otherwise touch.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import AdminUser, Folder
from tests.conftest import ADMIN_EMAIL
from tests.helpers import ensure_user, user_headers

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


def test_a_user_cannot_detach_a_video_from_a_folder_they_cannot_see(client):
    """The hole: `folder_id: null` was applied with no check on the source."""
    admin = _admin(client)
    folder = _private_folder(client, admin)
    video = _video_in(client, admin, folder["id"])
    headers = user_headers(ensure_user("detacher"))

    # The user cannot even read the folder, so the video is invisible to them.
    assert client.get(f"{API}/folders/{folder['id']}", headers=headers).status_code == 404

    resp = client.put(
        f"{API}/videos/{video['id']}/folder", json={"folder_id": None}, headers=headers
    )
    assert resp.status_code == 403, resp.text

    # The video is still filed where the admin put it.
    with SessionLocal() as db:
        assert db.get(__import__("app.models", fromlist=["Video"]).Video,
                      video["id"]).folder_id is not None


def test_a_user_cannot_move_a_video_into_a_folder_they_do_not_manage(client):
    admin = _admin(client)
    folder = _private_folder(client, admin)
    video = _video_in(client, admin, folder["id"])
    headers = user_headers(ensure_user("mover"))

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

    editor_email = ensure_user("cross-editor")
    with SessionLocal() as db:
        from app.models import FolderMember

        db.add(FolderMember(folder_id=target["id"], user_id=db.get(
            AdminUser, db.execute(select(AdminUser).where(
                AdminUser.email == editor_email)).scalar_one().id).id,
            role="editor"))
        db.commit()

    headers = user_headers(editor_email)
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
    headers = user_headers(ensure_user("patcher"))

    resp = client.patch(
        f"{API}/videos/{video['id']}",
        json={"folder_id": str(target["id"])},
        headers=headers,
    )
    assert resp.status_code == 403, resp.text

    # The user cannot edit a video in a folder they cannot read at all.
    resp = client.patch(
        f"{API}/videos/{video['id']}", json={"notes": "hello"}, headers=headers
    )
    assert resp.status_code == 403, resp.text


# ----------------------------------------------------------- unfiled media


def test_unfiled_media_is_not_served_to_a_user_who_is_not_the_creator(client):
    """Metadata 403s but media/audio used to return the bytes."""
    admin = _admin(client)
    video = _unfiled_video(client, admin)
    headers = user_headers(ensure_user("eavesdropper"))

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
    """A user must not read the queue that serves every folder."""
    admin = _admin(client)
    _private_folder(client, admin)
    headers = user_headers(ensure_user("queue-peeker"))

    for path in (f"{API}/jobs", f"{API}/stats"):
        resp = client.get(path, headers=headers)
        assert resp.status_code == 403, f"{path}: {resp.text}"
        assert client.get(path, headers=admin).status_code == 200, path


def test_a_user_cannot_mutate_the_operator_queue(client):
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

    headers = user_headers(ensure_user("job-tampering"))
    resp = client.post(f"{API}/jobs/{jobs[0]['id']}/cancel", headers=headers)
    assert resp.status_code == 403, resp.text


def test_merge_routes_require_the_video_role(client):
    """Merges are a video sub-resource and must inherit its role."""
    admin = _admin(client)
    folder = _private_folder(client, admin)
    video = _video_in(client, admin, folder["id"])
    headers = user_headers(ensure_user("merge-peeker"))

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


# ------------------------------------------------------------- the role tiers


def _role_headers(client, email: str, role: str, password: str = "role-pass-12345"):
    """Create an account with a role and a password, then log in as it.

    Roles are database-authoritative, so logging in normally is the honest way
    to mint each tier's token: whatever the JWT claims, the row decides.
    """
    from app.auth import hash_password

    with SessionLocal() as db:
        exists = db.execute(
            select(AdminUser).where(AdminUser.email == email)
        ).scalar_one_or_none()
        if exists is None:
            db.add(
                AdminUser(
                    email=email,
                    role=role,
                    password_hash=hash_password(password),
                    is_active=True,
                )
            )
            db.commit()
    resp = client.post(
        f"{API}/auth/login", json={"email": email, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _upload_unfiled(client, headers, *, title="Clip"):
    from tests.helpers import media_bytes

    resp = client.post(
        f"{API}/videos/upload",
        files={"file": ("clip.mp4", media_bytes(), "video/mp4")},
        data={"title": title, "start_transcription": "false"},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_a_developer_operates_the_pipeline_but_not_the_admin_surface(client):
    """Operator yes, admin tier no: the developer runs jobs, publishes nothing."""
    admin = _admin(client)
    dev = _role_headers(client, f"dev-{uuid4().hex[:8]}@test.local", "developer")

    # Operator: the queue and stats answer.
    assert client.get(f"{API}/jobs", headers=dev).status_code == 200
    assert client.get(f"{API}/stats", headers=dev).status_code == 200

    # Retry works on a video the developer has rights over (their own ingest).
    own = _upload_unfiled(client, dev, title="Dev clip")
    retried = client.post(f"{API}/videos/{own['id']}/retry", headers=dev)
    assert retried.status_code == 202, retried.text

    # Admin tier: publishing a public folder is above the developer.
    denied = client.post(
        f"{API}/folders", json={"name": "Public", "visibility": "public"}, headers=dev
    )
    assert denied.status_code == 403, denied.text
    # And so is editing someone else's transcript (that is the reviewer's job).
    admin = _admin(client)
    someone_elses = _upload_unfiled(client, admin, title="Not the dev's clip")
    resp = client.put(
        f"{API}/videos/{someone_elses['id']}/transcript",
        json={"content": "the developer cannot fix this"},
        headers=dev,
    )
    assert resp.status_code == 403, resp.text


def test_a_reviewer_edits_transcripts_but_not_the_pipeline(client):
    """Library yes, operator no: the reviewer reads everything and corrects
    transcripts anywhere, but touches neither the queue nor the media."""
    admin = _admin(client)
    reviewer = _role_headers(client, f"rev-{uuid4().hex[:8]}@test.local", "reviewer")

    # An unfiled video created by someone else is invisible to a plain user;
    # the reviewer may still correct its transcript, wherever it lives.
    unfiled = _upload_unfiled(client, admin, title="Needs review")
    edited = client.put(
        f"{API}/videos/{unfiled['id']}/transcript",
        json={"content": "human corrected"},
        headers=reviewer,
    )
    assert edited.status_code == 201, edited.text
    flagged = client.patch(
        f"{API}/videos/{unfiled['id']}/transcript",
        json={"cleaned": False},
        headers=reviewer,
    )
    assert flagged.status_code == 200, flagged.text

    # The reviewer's reach is the transcript, not the media: the listing no
    # longer shows a row they could not open at all (that unfiled unit would
    # 403 on `GET /videos/{id}`), while their own ingest is still listed.
    listing = client.get(f"{API}/videos", headers=reviewer).json()
    assert unfiled["id"] not in {v["id"] for v in listing["items"]}, listing
    own = _upload_unfiled(client, reviewer, title="Reviewer clip")
    listing = client.get(f"{API}/videos", headers=reviewer).json()
    assert own["id"] in {v["id"] for v in listing["items"]}, listing

    # But the operator surface and the media itself stay closed.
    assert client.get(f"{API}/jobs", headers=reviewer).status_code == 403
    assert client.delete(f"{API}/videos/{unfiled['id']}", headers=reviewer).status_code == 403


# ---------------------------------------------- the X-Requested-By closure


def test_a_requested_by_header_alone_is_not_a_credential(client):
    """The hole: ANY header value used to mint a principal, no token needed.

    `X-Requested-By: admin:anyone@x` with no bearer token must be the same as
    no token at all — a 401, not an admin.
    """
    for value in ("admin:anyone@x", "member:anyone@x", "telegram:123", "editor"):
        resp = client.get(
            f"{API}/auth/me", headers={"X-Requested-By": value}
        )
        assert resp.status_code == 401, f"{value}: {resp.text}"


def test_a_forged_requested_by_header_cannot_steal_an_identity(client):
    """The bot service token authenticates as the bot and nothing else.

    A non-`telegram` prefix next to the service token is not honored — the
    request stays the *service's*, so the header cannot be used to attribute
    the mutation to an invented admin account.
    """
    resp = client.get(
        f"{API}/auth/me",
        headers={
            "Authorization": f"Bearer {settings.bot_api_token}",
            "X-Requested-By": "admin:anyone@x",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["actor"] == "telegram:bot"
    assert body["role"] == "telegram"
    assert "anyone@x" not in body["actor"]

    # The bot's own form still works: the header names the Telegram user.
    resp = client.get(
        f"{API}/auth/me",
        headers={
            "Authorization": f"Bearer {settings.bot_api_token}",
            "X-Requested-By": "telegram:123",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["actor"] == "telegram:123"


def test_a_forged_requested_by_header_cannot_shadow_a_real_token(client, auth_headers):
    """An account token with a rogue header stays that account.

    The console never sends `X-Requested-By`, but a hostile client could
    attach one to a legitimate bearer token hoping to be re-attributed. The
    header is simply not honored there, so the request resolves to whoever
    the JWT actually belongs to.
    """
    resp = client.get(
        f"{API}/auth/me",
        headers={
            **auth_headers,
            "X-Requested-By": "admin:anyone-else@x",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["email"] == ADMIN_EMAIL
    assert body["role"] == "super_admin"
    assert body["actor"] == f"super_admin:{ADMIN_EMAIL}"
    assert "anyone-else" not in body["actor"]
