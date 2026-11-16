"""Per-segment language mix on dataset cards.

The portal's mix rail is only honest if it counts the tags Whisper actually
wrote per segment, so these tests store segments straight through the session
(the same shortcut `test_thumbnails.py` uses for a thumbnail URL) rather than
driving the transcription pipeline. A unit with no transcript, or a transcript
with no segments, must answer null rather than four zeros: those are different
facts and the rail renders them differently.
"""
from __future__ import annotations

from uuid import UUID

from app.db import SessionLocal
from app.models import Transcript, Video
from tests.helpers import make_folder

API = "/api/v1"


def _make_video(folder_id, *, title="Clip"):
    with SessionLocal() as db:
        video = Video(
            source_url=f"upload://{title}.mp4",
            platform="direct",
            title=title,
            status="transcribed",
            progress=100,
            folder_id=UUID(str(folder_id)),
            created_by="system",
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        return str(video.id)


def _add_transcript(video_id, segments):
    with SessionLocal() as db:
        db.add(
            Transcript(
                video_id=UUID(str(video_id)),
                version=1,
                source="whisper",
                language="km",
                content="mixed content",
                segments=segments,
                status="ready",
                created_by="system",
            )
        )
        db.commit()


def _card(client, headers, video_id):
    resp = client.get(
        f"{API}/dataset/cards", params={"video_id": video_id}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert len(items) == 1, items
    return items[0]


def test_counts_km_km_en_en_and_null(client, auth_headers):
    folder = make_folder(client, auth_headers)
    video = _make_video(folder["id"])
    _add_transcript(
        video,
        [
            {"start": 0.0, "end": 1.0, "text": "សួស្តី", "language": "km"},
            {"start": 1.0, "end": 2.0, "text": "ខ្ញុំ like this", "language": "km-en"},
            {"start": 2.0, "end": 3.0, "text": "hello", "language": "en"},
            {"start": 3.0, "end": 4.0, "text": "untagged", "language": None},
        ],
    )

    assert _card(client, auth_headers, video)["language_mix"] == {
        "km": 1,
        "km_en": 1,
        "en": 1,
        "und": 1,
    }


def test_transcript_without_segments_is_null(client, auth_headers):
    folder = make_folder(client, auth_headers)
    video = _make_video(folder["id"])
    _add_transcript(video, [])

    assert _card(client, auth_headers, video)["language_mix"] is None


def test_video_without_a_transcript_is_null(client, auth_headers):
    folder = make_folder(client, auth_headers)
    video = _make_video(folder["id"])

    assert _card(client, auth_headers, video)["language_mix"] is None


def test_unknown_or_malformed_segments_count_as_und(client, auth_headers):
    folder = make_folder(client, auth_headers)
    video = _make_video(folder["id"])
    _add_transcript(
        video,
        [
            {"text": "Thai from Whisper", "language": "th"},
            {"text": "no tag at all"},
            # Stored JSON from an older version is not guaranteed to be a list
            # of objects, and one odd row must not fail the whole listing.
            "not-an-object",
        ],
    )

    assert _card(client, auth_headers, video)["language_mix"] == {
        "km": 0,
        "km_en": 0,
        "en": 0,
        "und": 3,
    }


def test_public_folder_cards_carry_the_mix(client, auth_headers):
    folder = make_folder(client, auth_headers, visibility="public")
    video = _make_video(folder["id"])
    _add_transcript(
        video,
        [
            {"text": "សួស្តី", "language": "km"},
            {"text": "hello", "language": "en"},
        ],
    )

    resp = client.get(f"{API}/public/folders/{folder['id']}")

    assert resp.status_code == 200, resp.text
    cards = resp.json()["cards"]
    assert len(cards) == 1, cards
    assert cards[0]["language_mix"] == {"km": 1, "km_en": 0, "en": 1, "und": 0}
