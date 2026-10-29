"""Applying a merge promotes it as a NEW version of the current transcript.

Regression tests for F3: `POST /merges/{id}/apply` used to trust the
`merged_transcript_id` cached on the merge row (and could hand back that row),
so a second apply returned the same version, and the version it did create was
allocated without any protection against a concurrent writer for the video.
"""
from __future__ import annotations

import threading

from app.auth import Principal
from app.db import SessionLocal
from app.models import Job, Merge, Transcript
from app.routers.merges import apply_merge
from app.services.merger import merge_transcripts, segments_from_rows
from app.worker.runner import run_job

WHISPER_SEGMENTS = [
    {"start": 0.0, "end": 3.0, "text": "the birch can use lid on the smooth planks", "language": "en"},
    {"start": 3.0, "end": 6.0, "text": "glue the sheet to the dark blue background", "language": "en"},
    {"start": 6.0, "end": 9.0, "text": "rice is often served in round bowls", "language": "en"},
]

GEMINI_SEGMENTS = [
    {"start": 0.0, "end": 3.0, "text": "the birch canoe slid on the smooth planks", "language": "en"},
    {"start": 3.0, "end": 6.0, "text": "glue the sheet to the dark blue background", "language": "en"},
    {"start": 6.0, "end": 9.0, "text": "rice is often served in round bowls", "language": "en"},
]

PRINCIPAL = Principal(actor="admin@test.local", role="super_admin")


def _video(client, headers) -> str:
    resp = client.post(
        "/api/v1/videos",
        json={"source_url": "https://example.org/merge-apply-001"},
        headers=headers,
    )
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


def _ready_merge(base_id: str, gemini_id: str) -> str:
    """A merge row as the worker leaves it: ready, with a reviewed diff."""
    with SessionLocal() as db:
        base = db.get(Transcript, base_id)
        gemini = db.get(Transcript, gemini_id)
        result = merge_transcripts(
            segments_from_rows(base.segments), segments_from_rows(gemini.segments)
        )
        row = Merge(
            video_id=base.video_id,
            base_transcript_id=base.id,
            gemini_transcript_id=gemini.id,
            status="ready",
            model="gemini-2.5-pro",
            diff={
                "segments": result.segments,
                "summary": result.summary,
                "stats": result.stats,
            },
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return str(row.id)


def _run_queued_merge_job() -> None:
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


def test_apply_promotes_the_merge_as_a_new_version_on_top_of_current_state(
    client, auth_headers
):
    video_id = _video(client, auth_headers)
    _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)
    _add_transcript(video_id, 2, "gemini", GEMINI_SEGMENTS)

    created = client.post(
        f"/api/v1/videos/{video_id}/transcript/merge", json={}, headers=auth_headers
    )
    assert created.status_code == 202, created.text
    _run_queued_merge_job()

    applied = client.post(f"/api/v1/merges/{created.json()['id']}/apply", headers=auth_headers)
    assert applied.status_code == 200, applied.text
    body = applied.json()
    assert body["source"] == "merged"
    assert body["version"] == 3  # v1 whisper, v2 gemini -> next is 3
    reviewed = client.get(f"/api/v1/merges/{created.json()['id']}", headers=auth_headers)
    diff_text = " ".join(s["text"] for s in reviewed.json()["diff"]["segments"])
    assert body["content"] == diff_text, "apply must promote the reviewed diff verbatim"

    current = client.get(f"/api/v1/videos/{video_id}/transcript", headers=auth_headers)
    assert current.json()["version"] == 3


def test_apply_reacts_to_a_transcript_written_after_the_merge_was_ready(
    client, auth_headers
):
    """The version is recomputed at apply time, not cached on the merge row."""
    video_id = _video(client, auth_headers)
    whisper_id = _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)
    gemini_id = _add_transcript(video_id, 2, "gemini", GEMINI_SEGMENTS)
    merge_id = _ready_merge(whisper_id, gemini_id)

    # A human corrects the transcript while the merge sits ready in review.
    _add_transcript(video_id, 3, "manual", GEMINI_SEGMENTS)

    applied = client.post(f"/api/v1/merges/{merge_id}/apply", headers=auth_headers)
    assert applied.status_code == 200, applied.text
    assert applied.json()["version"] == 4, "apply must allocate on top of the current state"


def test_applying_twice_yields_two_distinct_increasing_versions(client, auth_headers):
    video_id = _video(client, auth_headers)
    whisper_id = _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)
    gemini_id = _add_transcript(video_id, 2, "gemini", GEMINI_SEGMENTS)
    merge_id = _ready_merge(whisper_id, gemini_id)

    first = client.post(f"/api/v1/merges/{merge_id}/apply", headers=auth_headers)
    second = client.post(f"/api/v1/merges/{merge_id}/apply", headers=auth_headers)
    assert first.status_code == second.status_code == 200
    versions = [first.json()["version"], second.json()["version"]]
    assert versions == [3, 4], versions
    assert len(set(versions)) == 2

    with SessionLocal() as db:
        rows = (
            db.query(Transcript)
            .filter(Transcript.video_id == video_id, Transcript.source == "merged")
            .order_by(Transcript.version)
            .all()
        )
        assert [r.version for r in rows] == [3, 4]


def test_apply_never_hardcodes_version_one(client, auth_headers):
    video_id = _video(client, auth_headers)
    whisper_id = _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)
    gemini_id = _add_transcript(video_id, 2, "gemini", GEMINI_SEGMENTS)
    # Older history the merge knows nothing about.
    for version in range(3, 8):
        _add_transcript(video_id, version, "manual", GEMINI_SEGMENTS)
    merge_id = _ready_merge(whisper_id, gemini_id)

    applied = client.post(f"/api/v1/merges/{merge_id}/apply", headers=auth_headers)
    assert applied.status_code == 200, applied.text
    assert applied.json()["version"] == 8


def test_concurrent_applies_cannot_allocate_the_same_version(client, auth_headers):
    """Two simultaneous applies of one merge must not both write version 3."""
    video_id = _video(client, auth_headers)
    whisper_id = _add_transcript(video_id, 1, "whisper", WHISPER_SEGMENTS)
    gemini_id = _add_transcript(video_id, 2, "gemini", GEMINI_SEGMENTS)
    merge_id = _ready_merge(whisper_id, gemini_id)

    results: list[tuple[int, int]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def _apply() -> None:
        try:
            barrier.wait(timeout=10)
            with SessionLocal() as db:
                row = apply_merge(merge_id, db=db, principal=PRINCIPAL)
                results.append((row.version, 0))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion
            errors.append(exc)

    threads = [threading.Thread(target=_apply) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert not any(t.is_alive() for t in threads), "apply deadlocked"
    assert not errors, errors

    versions = sorted(version for version, _ in results)
    assert versions == [3, 4], versions

    with SessionLocal() as db:
        stored = sorted(
            row.version
            for row in db.query(Transcript).filter(
                Transcript.video_id == video_id, Transcript.source == "merged"
            )
        )
    assert stored == [3, 4], stored
