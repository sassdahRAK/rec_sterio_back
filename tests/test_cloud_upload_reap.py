"""Reclaiming abandoned personal-cloud uploads.

`POST /cloud/uploads` commits the Video row (status `uploading`) before any
bytes exist; the client PUTs to the provider and calls `/complete` separately.
If that never happens the row used to sit `uploading` forever. The worker's
poll loop now sweeps it with `queue.reap_stale_cloud_uploads`.
"""
from __future__ import annotations

import inspect
from datetime import timedelta

import pytest

from app.config import settings
from app.db import SessionLocal
from app.models import Video
from app.worker import main as worker_main
from app.worker import queue


def _make_video(**overrides) -> Video:
    """Insert one Video row; defaults describe a stale cloud upload."""
    values = {
        "source_url": "upload://clip.mp4",
        "platform": "direct",
        "status": "uploading",
        "progress": 42,
        "created_by": "user:reap-test",
        "storage_backend": "google",
        "created_at": queue.utcnow() - timedelta(days=2),
    }
    values.update(overrides)
    with SessionLocal() as db:
        video = Video(**values)
        db.add(video)
        db.commit()
        return video


def _reap() -> int:
    with SessionLocal() as db:
        return queue.reap_stale_cloud_uploads(db)


def _reload(video: Video) -> Video:
    with SessionLocal() as db:
        current = db.get(Video, video.id)
        # Detach the values we assert on before the session closes.
        db.expunge(current)
        return current


def test_stale_uploading_cloud_row_becomes_failed():
    video = _make_video()

    assert _reap() == 1

    current = _reload(video)
    assert current.status == "failed"
    assert current.progress == 0
    assert current.error == queue.ABANDONED_CLOUD_UPLOAD_ERROR
    # The message is rendered verbatim by the portal: no dash characters.
    assert "—" not in current.error
    assert "–" not in current.error
    assert " - " not in current.error


def test_fresh_uploading_row_is_untouched():
    video = _make_video(
        created_at=queue.utcnow()
        - timedelta(minutes=max(1, settings.cloud_upload_ttl_minutes // 2))
    )

    assert _reap() == 0

    current = _reload(video)
    assert current.status == "uploading"
    assert current.progress == 42
    assert current.error is None


def test_uploading_row_with_drive_file_id_is_untouched():
    """A file id means the bytes landed: `complete` is mid-flight."""
    video = _make_video(drive_file_id="drive-file-abc")

    assert _reap() == 0

    current = _reload(video)
    assert current.status == "uploading"
    assert current.error is None


def test_local_row_is_untouched():
    video = _make_video(storage_backend=None)

    assert _reap() == 0

    current = _reload(video)
    assert current.status == "uploading"
    assert current.error is None


def test_second_pass_is_idempotent():
    _make_video()

    assert _reap() == 1
    assert _reap() == 0


def test_failed_row_is_not_retouched():
    video = _make_video(status="failed", error="something else went wrong")

    assert _reap() == 0

    current = _reload(video)
    assert current.status == "failed"
    assert current.error == "something else went wrong"


def test_fresh_rows_survive_next_to_stale_ones():
    stale = _make_video()
    fresh = _make_video(created_at=queue.utcnow() - timedelta(minutes=5))

    assert _reap() == 1

    assert _reload(stale).status == "failed"
    assert _reload(fresh).status == "uploading"


def test_worker_loop_calls_the_sweep():
    """Source-level check: invoking the real loop needs threads and a DB poll."""
    loop_src = inspect.getsource(worker_main.worker_loop)
    assert "_maybe_reap_stale_uploads" in loop_src
    hook_src = inspect.getsource(worker_main._maybe_reap_stale_uploads)
    assert "reap_stale_cloud_uploads" in hook_src


def test_sweep_exception_does_not_propagate(monkeypatch):
    monkeypatch.setattr(worker_main, "_last_reap", 0.0)

    def _boom(db):  # noqa: ARG001 - signature of the real sweep
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(queue, "reap_stale_cloud_uploads", _boom)

    worker_main._maybe_reap_stale_uploads()  # must not raise

    assert worker_main._last_reap > 0.0


def test_sweep_is_gated_between_polls(monkeypatch):
    monkeypatch.setattr(worker_main, "_last_reap", 0.0)
    calls: list[int] = []

    def _count(db):  # noqa: ARG001
        calls.append(1)
        return 0

    monkeypatch.setattr(queue, "reap_stale_cloud_uploads", _count)

    worker_main._maybe_reap_stale_uploads()
    worker_main._maybe_reap_stale_uploads()

    assert len(calls) == 1


@pytest.fixture(autouse=True)
def _reset_reap_clock():
    """Never let a previous test's timestamp gate the next one."""
    yield
    worker_main._last_reap = 0.0
