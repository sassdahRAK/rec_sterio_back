"""Merge API: resolving the base transcript (regression tests).

A merge started without explicit transcript ids must pair the *primary*
transcript (Whisper / manual / merged) with the Gemini second opinion. It must
never pick the Gemini row as the base, which would compare Gemini with itself
and silently drop everything only Whisper heard.
"""
from __future__ import annotations

from app.db import SessionLocal
from app.models import Job, Merge, Transcript
from app.worker.runner import run_job

URL = "https://www.youtube.com/watch?v=recsterio-merge-api-001"

WHISPER_SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "the birch can use lid on the smooth planks", "language": "en"},
    {"start": 4.0, "end": 7.0, "text": "glue the sheet to the dark blue background", "language": "en"},
    {"start": 7.0, "end": 10.0, "text": "it is easy to tell the depth of a well", "language": "en"},
    {"start": 10.0, "end": 14.0, "text": "these days a chicken leg is a rare dish", "language": "en"},
    {"start": 14.0, "end": 18.0, "text": "rice is often served in round bowls", "language": "en"},
    {"start": 18.0, "end": 22.0, "text": "the juice of lemons makes fine punch", "language": "en"},
    {"start": 22.0, "end": 25.0, "text": "a whisker is an instrument for measuring weight", "language": "en"},
    {"start": 25.0, "end": 28.0, "text": "the tree was a hundred feet tall", "language": "en"},
    {"start": 28.0, "end": 31.0, "text": "a frog hops across the road", "language": "en"},
    {"start": 31.0, "end": 34.0, "text": "the ship sails tomorrow morning", "language": "en"},
]

GEMINI_SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "the birch canoe slid on the smooth planks", "language": "en"},
    {"start": 4.0, "end": 7.0, "text": "glue the sheet to the dark blue background", "language": "en"},
    {"start": 7.0, "end": 10.0, "text": "it is easy to tell the depth of a well", "language": "en"},
    {"start": 10.0, "end": 14.0, "text": "these days a chicken leg is a rare dish", "language": "en"},
    {"start": 14.0, "end": 18.0, "text": "rice is often served in round bowls", "language": "en"},
    {"start": 18.0, "end": 22.0, "text": "the juice of lemons makes fine punch", "language": "en"},
]


def _create_video(client, headers) -> str:
    resp = client.post("/api/v1/videos", json={"source_url": URL}, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _add_transcript(video_id: str, version: int, source: str, segments: list[dict]) -> str:
    with SessionLocal() as db:
        row = Transcript(
            video_id=video_id,
            version=version,
            source=source,
            language="en",
            content=" ".join(s["text"] for s in segments),
            segments=segments,
            status="ready",
            created_by=source,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return str(row.id)


def _run_pending_merge_job() -> None:
    with SessionLocal() as db:
        job = (
            db.query(Job)
            .filter(Job.kind == "merge", Job.status == "queued")
            .order_by(Job.created_at.desc())
            .first()
        )
        assert job is not None, "expected an enqueued merge job"
        run_job(db, job)
        assert job.status == "succeeded", job.last_error


def test_merge_without_ids_uses_whisper_base_and_keeps_every_segment(client, auth_headers):
    video_id = _create_video(client, auth_headers)
    whisper_id = _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)
    gemini_id = _add_transcript(video_id, 2, "gemini", GEMINI_SEGMENTS)
    # An earlier, already-merged version must not shadow the independent
    # Whisper pass when the base is resolved automatically.
    _add_transcript(video_id, 3, "merged", GEMINI_SEGMENTS)

    created = client.post(
        f"/api/v1/videos/{video_id}/transcript/merge", json={}, headers=auth_headers
    )
    assert created.status_code == 202, created.text
    merge = created.json()
    assert merge["base_transcript_id"] == whisper_id, "base must be the Whisper transcript"
    assert merge["gemini_transcript_id"] is None

    _run_pending_merge_job()

    with SessionLocal() as db:
        merge_row = db.get(Merge, merge["id"])
        assert merge_row.status == "ready", merge_row.error
        assert str(merge_row.base_transcript_id) == whisper_id
        assert str(merge_row.gemini_transcript_id) == gemini_id

        stats = merge_row.diff["stats"]
        assert stats["base_segments"] == len(WHISPER_SEGMENTS)
        assert stats["output_segments"] >= len(WHISPER_SEGMENTS)
        assert stats["base_only"] == len(WHISPER_SEGMENTS) - len(GEMINI_SEGMENTS)

        merged_text = " ".join(s["text"] for s in merge_row.diff["segments"])
        # Nothing that only Whisper heard may be dropped.
        for segment in WHISPER_SEGMENTS:
            assert segment["text"] in merged_text


def test_apply_promotes_the_merged_version(client, auth_headers):
    video_id = _create_video(client, auth_headers)
    _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)
    _add_transcript(video_id, 2, "gemini", GEMINI_SEGMENTS)

    merge = client.post(
        f"/api/v1/videos/{video_id}/transcript/merge", json={}, headers=auth_headers
    ).json()
    _run_pending_merge_job()

    applied = client.post(f"/api/v1/merges/{merge['id']}/apply", headers=auth_headers)
    assert applied.status_code == 200, applied.text
    assert applied.json()["source"] == "merged"
    assert applied.json()["version"] == 3

    current = client.get(f"/api/v1/videos/{video_id}/transcript", headers=auth_headers).json()
    assert current["version"] == 3 and current["source"] == "merged"


def test_merge_with_only_a_gemini_transcript_is_404(client, auth_headers):
    video_id = _create_video(client, auth_headers)
    _add_transcript(video_id, 1, "gemini", GEMINI_SEGMENTS)

    resp = client.post(
        f"/api/v1/videos/{video_id}/transcript/merge", json={}, headers=auth_headers
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"


def test_merge_with_same_transcript_twice_is_409(client, auth_headers):
    video_id = _create_video(client, auth_headers)
    whisper_id = _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)

    resp = client.post(
        f"/api/v1/videos/{video_id}/transcript/merge",
        json={"base_transcript_id": whisper_id, "gemini_transcript_id": whisper_id},
        headers=auth_headers,
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "conflict"
