"""Forking a card into your own folder (`POST /videos/{id}/copy`).

The distinction these tests pin is move-versus-copy. A move is curation: it
takes a unit out of wherever it was and needs edit rights there. A copy is
perusal: it needs only read on the source, leaves the original alone, and is
the only way a user can do anything with a colleague's public unit.

The last two tests cover the side effect that made copying interesting: a fork
puts a private duplicate of a shared URL in the database, so the ingestion
de-duplication lookups must not answer with somebody else's private row.
"""
from __future__ import annotations

import os
from uuid import UUID

from app.db import SessionLocal
from app.models import Video
from app.services import storage
from tests.helpers import (
    ensure_user,
    make_folder,
    make_transcript,
    user_headers,
    scoped_headers,
    upload_video,
)

API = "/api/v1"
LINK = "https://www.youtube.com/watch?v=recsterio-copy-test-001"


def _user_folder(client, tag: str = "forker") -> tuple[dict, str, dict]:
    """A signed-in user who owns a private folder of their own."""
    email = ensure_user(tag)
    headers = user_headers(email)
    folder = make_folder(client, headers, name=f"{tag} copies")
    return folder, email, headers


def _public_card(client, auth_headers, *, content="សួស្តី hello", cleaned=True):
    public = make_folder(client, auth_headers, name="Public voices", visibility="public")
    video = upload_video(client, auth_headers, folder_id=public["id"], title="Unit A")
    make_transcript(client, auth_headers, video["id"], content, cleaned=cleaned)
    return public, video


def _attach_media(video_id: str) -> None:
    """Give a link-created (otherwise media-less) video a file on disk.

    `create_video` only queues a download; no worker runs in tests. Attaching
    the bytes directly is what lets the copy path — which requires real media —
    be exercised without a network fetch.
    """
    path = storage.video_dir(video_id) / "media.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(32))
    with SessionLocal() as db:
        video = db.get(Video, UUID(video_id))
        video.local_path = str(path)
        video.status = "downloaded"
        video.progress = 100
        db.commit()


# ------------------------------------------------------------------ the happy path


def test_user_forks_a_public_card_into_their_own_folder(client, auth_headers):
    public, source = _public_card(client, auth_headers)
    mine, email, user = _user_folder(client)

    resp = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    )
    assert resp.status_code == 201, resp.text
    fork = resp.json()

    assert fork["copied_from_id"] == source["id"]
    assert fork["folder_id"] == mine["id"]
    assert fork["status"] == "transcribed"
    assert fork["created_by"] == f"user:{email}"
    assert fork["title"] == "Unit A"

    # The original is untouched: still in the public folder, still public.
    original = client.get(f"{API}/videos/{source['id']}", headers=auth_headers).json()
    assert original["folder_id"] == public["id"]
    assert original["copied_from_id"] is None


def test_copy_appears_as_a_card_with_provenance(client, auth_headers):
    _, source = _public_card(client, auth_headers)
    mine, _, user = _user_folder(client)
    client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    )

    cards = client.get(
        f"{API}/dataset/cards", params={"folder_id": mine["id"]}, headers=user
    ).json()["items"]
    assert len(cards) == 1
    assert cards[0]["copied_from_id"] == source["id"]
    assert cards[0]["badge"] == "cleaned"
    assert cards[0]["folder_name"] == mine["name"]


def test_copy_appends_to_the_end_of_the_destination(client, auth_headers):
    _, source = _public_card(client, auth_headers)
    mine, _, user = _user_folder(client)
    first = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    ).json()
    second = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    ).json()

    assert [first["position"], second["position"]] == [0, 1]
    # Two forks are independent rows, not the same row twice.
    assert first["id"] != second["id"]


# ------------------------------------------------------------------ the snapshot


def test_copy_clones_the_transcript_as_an_independent_version(client, auth_headers):
    _, source = _public_card(client, auth_headers, content="original text", cleaned=True)
    mine, _, user = _user_folder(client)
    fork = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    ).json()

    cloned = client.get(f"{API}/videos/{fork['id']}/transcript", headers=user).json()
    assert cloned["content"] == "original text"
    assert cloned["cleaned"] is True
    assert cloned["version"] == 1

    # Editing the copy must never reach the original.
    client.put(
        f"{API}/videos/{fork['id']}/transcript",
        json={"content": "my own corrections", "cleaned": True},
        headers=user,
    )
    still = client.get(f"{API}/videos/{source['id']}/transcript", headers=auth_headers).json()
    assert still["content"] == "original text"


def test_copy_does_not_duplicate_the_tvna_serial(client, auth_headers):
    _, source = _public_card(client, auth_headers)
    recorded = client.post(
        f"{API}/videos/{source['id']}/tvna",
        json={"location_tag": "Wat Phnom", "collected_by": "Dara"},
        headers=auth_headers,
    ).json()
    assert recorded["tvna_serial"] == 1

    mine, _, user = _user_folder(client)
    fork = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    ).json()

    # The collection fields describe the item, so they travel with the copy;
    # the serial names the original and stays unique.
    assert fork["tvna_serial"] is None
    assert fork["location_tag"] == "Wat Phnom"
    assert fork["collected_by"] == "Dara"

    # Recording the copy is a deliberate act and takes the next number.
    again = client.post(
        f"{API}/videos/{fork['id']}/tvna", json={"location_tag": "Wat Phnom"}, headers=user
    ).json()
    assert again["tvna_serial"] == 2


def test_copy_media_is_shared_and_outlives_the_original(client, auth_headers):
    _, source = _public_card(client, auth_headers)
    mine, _, user = _user_folder(client)
    fork = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    ).json()

    source_media = storage.find_media_file(source["id"])
    fork_media = storage.find_media_file(fork["id"])
    assert source_media is not None and fork_media is not None
    assert source_media.read_bytes() == fork_media.read_bytes()
    # Hard links: one inode, no second download and no second copy on disk.
    assert os.stat(source_media).st_ino == os.stat(fork_media).st_ino

    assert client.delete(f"{API}/videos/{source['id']}", headers=auth_headers).status_code == 204

    playable = client.get(f"{API}/videos/{fork['id']}/media", headers=user)
    assert playable.status_code == 200
    assert playable.content == fork_media.read_bytes()
    # Deleting the original clears the provenance pointer but not the copy.
    assert client.get(f"{API}/videos/{fork['id']}", headers=user).json()["copied_from_id"] is None


# ------------------------------------------------------------------ the guards


def test_copy_requires_read_on_the_source(client, auth_headers):
    theirs = make_folder(client, auth_headers, name="Their private notes")
    source = upload_video(client, auth_headers, folder_id=theirs["id"])
    mine, _, user = _user_folder(client)

    resp = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"


def test_copy_requires_edit_on_the_destination(client, auth_headers):
    _, source = _public_card(client, auth_headers)
    theirs = make_folder(client, auth_headers, name="Read-only for me")
    _, _, user = _user_folder(client)

    resp = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": theirs["id"]}, headers=user
    )
    assert resp.status_code in (403, 404), resp.text


def test_copy_into_the_same_folder_is_rejected(client, auth_headers):
    public, source = _public_card(client, auth_headers)
    resp = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": public["id"]}, headers=auth_headers
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["details"][0]["issue"] == "same_folder"


def test_copy_of_a_video_without_media_is_not_ready(client, auth_headers):
    public = make_folder(client, auth_headers, name="Public links", visibility="public")
    created = client.post(
        f"{API}/videos", json={"source_url": LINK, "folder_id": public["id"]}, headers=auth_headers
    )
    assert created.status_code == 201, created.text
    mine, _, user = _user_folder(client)

    resp = client.post(
        f"{API}/videos/{created.json()['id']}/copy",
        json={"folder_id": mine["id"]},
        headers=user,
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "not_ready"


def test_a_passcode_visitor_cannot_fork(client, auth_headers):
    """Forking a passcode-opened folder would defeat the passcode's expiry."""
    locked = make_folder(client, auth_headers, name="Passcode only")
    source = upload_video(client, auth_headers, folder_id=locked["id"])
    visitor = scoped_headers(locked["id"], role="viewer")

    resp = client.post(
        f"{API}/videos/{source['id']}/copy", json={"folder_id": locked["id"]}, headers=visitor
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "forbidden"


# ------------------------------------------------------------------ de-duplication


def test_a_private_fork_does_not_shadow_the_link_for_others(client, auth_headers):
    """The fork is private; re-submitting the link must not report its id."""
    public = make_folder(client, auth_headers, name="Public links", visibility="public")
    created = client.post(
        f"{API}/videos", json={"source_url": LINK, "folder_id": public["id"]}, headers=auth_headers
    ).json()
    _attach_media(created["id"])

    mine, _, user = _user_folder(client)
    fork = client.post(
        f"{API}/videos/{created['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    ).json()
    assert fork["status"] == "downloaded"

    # The original disappears; only the private fork remains.
    assert client.delete(f"{API}/videos/{created['id']}", headers=auth_headers).status_code == 204

    other_email = ensure_user("outsider")
    other = user_headers(other_email)
    resp = client.post(f"{API}/videos", json={"source_url": LINK}, headers=other)
    assert resp.status_code == 201, resp.text
    assert resp.json()["created_by"] == f"user:{other_email}"


def test_a_fork_does_not_shadow_the_original_while_it_exists(client, auth_headers):
    """The canonical row is still the one an ingest collision reports."""
    public = make_folder(client, auth_headers, name="Public links", visibility="public")
    created = client.post(
        f"{API}/videos", json={"source_url": LINK, "folder_id": public["id"]}, headers=auth_headers
    ).json()
    _attach_media(created["id"])
    mine, _, user = _user_folder(client)
    client.post(
        f"{API}/videos/{created['id']}/copy", json={"folder_id": mine["id"]}, headers=user
    )

    resp = client.post(f"{API}/videos", json={"source_url": LINK}, headers=user)
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["details"][0]["video_id"] == created["id"]


def test_an_unfiled_upload_can_be_filed_into_its_creators_folder(client, auth_headers):
    """Unfiled means "only its uploader can see it", so filing is a move they own."""
    mine, email, user = _user_folder(client)
    unfiled = upload_video(client, user, title="Mine, unfiled")
    assert unfiled["folder_id"] is None

    resp = client.put(
        f"{API}/videos/{unfiled['id']}/folder", json={"folder_id": mine["id"]}, headers=user
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["folder_id"] == mine["id"]
    assert resp.json()["created_by"] == f"user:{email}"
