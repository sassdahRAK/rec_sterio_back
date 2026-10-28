"""TvNa collection schema (instruction.txt #6) and the Telegram mirror (#4)."""
from __future__ import annotations

from app.db import SessionLocal
from app.models import Video
from app.services import downloader

URL = "https://www.tiktok.com/@chewbash/video/7000000000000000001"


def make_video(client, headers, url: str = URL) -> dict:
    resp = client.post("/api/v1/videos", json={"source_url": url}, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ------------------------------------------------------------------ TvNa record


def test_recording_tvna_assigns_a_sequential_id(client, auth_headers):
    video = make_video(client, auth_headers)
    resp = client.post(
        f"/api/v1/videos/{video['id']}/tvna",
        json={"location_tag": "Chew & Bash, Phnom Penh"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tvna_serial"] == 1
    assert body["tvna_id"] == "PP-001"
    assert body["location_tag"] == "Chew & Bash, Phnom Penh"
    # Defaults the collector does not have to type.
    assert body["source_type"] == "Video"
    assert body["collection_method"] == "Manual browse"
    assert body["collected_by"] == "admin:admin@test.local"
    assert body["collection_date"]


def test_sequential_ids_do_not_collide(client, auth_headers):
    ids = []
    for n in range(3):
        video = make_video(client, auth_headers, url=f"{URL}?n={n}")
        resp = client.post(
            f"/api/v1/videos/{video['id']}/tvna",
            json={"location_tag": f"Place {n}"},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        ids.append(resp.json()["tvna_id"])
    assert ids == ["PP-001", "PP-002", "PP-003"]


def test_recording_twice_keeps_the_same_number(client, auth_headers):
    video = make_video(client, auth_headers)
    first = client.post(
        f"/api/v1/videos/{video['id']}/tvna",
        json={"location_tag": "Original place"},
        headers=auth_headers,
    ).json()
    second = client.post(
        f"/api/v1/videos/{video['id']}/tvna",
        json={"location_tag": "Corrected place", "notes": "verified on site"},
        headers=auth_headers,
    ).json()

    # The PP number is what appears in the spreadsheet and the report; a
    # correction must not renumber the item.
    assert second["tvna_id"] == first["tvna_id"]
    assert second["location_tag"] == "Corrected place"
    assert second["notes"] == "verified on site"


def test_location_tag_is_required(client, auth_headers):
    video = make_video(client, auth_headers)
    resp = client.post(
        f"/api/v1/videos/{video['id']}/tvna", json={}, headers=auth_headers
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_failed"


def test_unknown_source_type_is_rejected(client, auth_headers):
    video = make_video(client, auth_headers)
    resp = client.post(
        f"/api/v1/videos/{video['id']}/tvna",
        json={"location_tag": "Place", "source_type": "Podcast"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_tvna_requires_edit_access(client):
    """An anonymous caller must not be able to record collection data."""
    resp = client.post(
        "/api/v1/videos/00000000-0000-0000-0000-000000000000/tvna",
        json={"location_tag": "Nowhere"},
    )
    assert resp.status_code in (401, 403, 404)


def test_detected_metadata_is_kept_when_the_collector_omits_it(client, auth_headers):
    video = make_video(client, auth_headers)
    # Simulate what the downloader fills in after a real fetch.
    with SessionLocal() as db:
        row = db.get(Video, video["id"])
        row.author_handle = "@chewbash"
        row.caption = "Chew & Bash is the best late-night spot in Phnom Penh"
        row.date_posted = "2026-09-14"
        db.commit()

    body = client.post(
        f"/api/v1/videos/{video['id']}/tvna",
        json={"location_tag": "Chew & Bash, Phnom Penh"},
        headers=auth_headers,
    ).json()
    assert body["author_handle"] == "@chewbash"
    assert body["date_posted"] == "2026-09-14"
    assert body["caption"].startswith("Chew & Bash is")


def test_collector_can_override_detected_metadata(client, auth_headers):
    video = make_video(client, auth_headers)
    body = client.post(
        f"/api/v1/videos/{video['id']}/tvna",
        json={
            "location_tag": "Chew & Bash",
            "author_handle": "@someone-else",
            "date_posted": "n.d.",
            "collected_by": "Omra",
        },
        headers=auth_headers,
    ).json()
    assert body["author_handle"] == "@someone-else"
    assert body["date_posted"] == "n.d."
    assert body["collected_by"] == "Omra"


def test_unrecorded_video_has_no_tvna_id(client, auth_headers):
    video = make_video(client, auth_headers)
    assert video["tvna_id"] is None
    assert video["tvna_serial"] is None


# --------------------------------------------------------------- Telegram mirror


def test_telegram_copy_is_recorded(client, auth_headers):
    video = make_video(client, auth_headers)
    resp = client.post(
        f"/api/v1/videos/{video['id']}/telegram",
        json={"file_id": "BQACAgQAAxkBAAI", "message_id": 42},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["telegram_file_id"] == "BQACAgQAAxkBAAI"
    assert body["telegram_message_id"] == 42


def test_telegram_link_rejects_an_empty_file_id(client, auth_headers):
    video = make_video(client, auth_headers)
    resp = client.post(
        f"/api/v1/videos/{video['id']}/telegram",
        json={"file_id": ""},
        headers=auth_headers,
    )
    assert resp.status_code == 422


# ----------------------------------------------------- downloader metadata helpers


def test_handle_is_prefixed_with_at():
    assert downloader.normalize_handle({"uploader_id": "chewbash"}) == "@chewbash"
    assert downloader.normalize_handle({"uploader_id": "@chewbash"}) == "@chewbash"
    assert downloader.normalize_handle({"uploader": "Chew & Bash"}) == "Chew & Bash"
    assert downloader.normalize_handle({}) is None


def test_caption_is_the_first_twenty_words_verbatim():
    text = " ".join(f"word{i}" for i in range(40))
    result = downloader.first_words(text)
    assert result == " ".join(f"word{i}" for i in range(20))
    # Verbatim: no ellipsis appended.
    assert not result.endswith("…")
    assert downloader.first_words(None) is None
    assert downloader.first_words("   ") is None


def test_upload_date_is_normalised():
    assert downloader.normalize_upload_date("20260914") == "2026-09-14"
    assert downloader.normalize_upload_date("2026-09-14") == "2026-09-14"
    assert downloader.normalize_upload_date(None) == "n.d."


def test_blocked_platforms_get_an_actionable_hint():
    tiktok = downloader._block_hint("tiktok")
    assert "YTDLP_COOKIES_FILE" in tiktok
    facebook = downloader._block_hint("facebook")
    assert "c_user" in facebook
    # An unknown platform still names the remedy rather than shrugging.
    assert "YTDLP_COOKIES_FILE" in downloader._block_hint("unknown")


# --------------------------------------------------- blocked-platform error path


def test_a_tiktok_block_becomes_an_actionable_permanent_error(monkeypatch):
    """yt-dlp's opaque message must reach the user as advice, not noise."""
    import pytest

    class _Ydl:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            raise RuntimeError(
                "Unexpected response from webpage request for "
                "https://www.tiktok.com/@x/video/1"
            )

    monkeypatch.setattr(downloader, "_build_ydl", lambda opts: _Ydl())

    with pytest.raises(downloader.DownloadError) as caught:
        downloader.download("https://www.tiktok.com/@x/video/1", "00000000-0000-0000-0000-000000000000")

    message = str(caught.value)
    assert "blocked this request" in message
    # The remedy, not "try again".
    assert "YTDLP_COOKIES_FILE" in message


def test_a_generic_failure_stays_transient(monkeypatch):
    """A network blip must be retryable — only a bot check is permanent."""
    import pytest

    class _Ydl:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            raise RuntimeError("Connection reset by peer")

    monkeypatch.setattr(downloader, "_build_ydl", lambda opts: _Ydl())

    with pytest.raises(downloader.TransientDownloadError):
        downloader.download("https://example.com/a.mp4", "00000000-0000-0000-0000-000000000000")
