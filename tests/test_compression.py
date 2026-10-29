"""P3 delivery compression: ffmpeg service, worker handler, media variant.

The plan's explicit gate lives here: compression is delivery-only and must
never become an input to the ASR pipeline (`test_compression_never_feeds_asr`).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.config import settings
from app.db import SessionLocal
from app.models import Job, Video
from app.services import asr as asr_service
from app.services import storage
from app.worker import queue
from app.worker.runner import compress_plan
from app.worker.runner import run_job
from tests.helpers import (
    ensure_user,
    guest_headers,
    large_media_file,
    make_folder,
    upload_video,
    user_headers,
)

API = "/api/v1"
MIB = 1024 * 1024


# ------------------------------------------------------------------ helpers


def _video_with_media(size: int = 16) -> str:
    """A video row plus a real `media.mp4` of `size` bytes (no download job)."""
    with SessionLocal() as db:
        video = Video(
            source_url=f"https://example.com/compress-{uuid4().hex}",
            platform="youtube",
            status="downloaded",
            progress=100,
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        video_id = str(video.id)
    media = storage.media_file(video_id, "mp4")
    media.write_bytes(b"x" * size)
    return video_id


def _run_compress_job(video_id: str) -> Job:
    with SessionLocal() as db:
        job = queue.enqueue(db, "compress_video", video_id=video_id)
        db.commit()
        job_id = str(job.id)
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        run_job(db, job)
        db.refresh(job)
        return job


def _recording_compress(sizes: dict, calls: list):
    """A fake rung encoder; `sizes` maps (height, crf) to an output byte size."""

    def fake(media_path, out_path, *, height, crf, maxrate_kbps=None, timeout_seconds=None):
        calls.append(
            {
                "height": height,
                "crf": crf,
                "maxrate_kbps": maxrate_kbps,
                "timeout_seconds": timeout_seconds,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\x00" * sizes[(height, crf)])
        return out_path

    return fake


# ------------------------------------------------------------------ chooser


def test_plan_skips_originals_at_or_under_the_threshold():
    """<=50 MB: the bot sends the original, so there is no ladder at all."""
    assert compress_plan(50 * MIB, 600.0, 1080, threshold_mb=50) == []
    assert compress_plan(1024, 10.0, 720, threshold_mb=50) == []


def test_plan_orders_the_ladder_and_scales_only_downward():
    """1080p scales to every rung; a 360p source is never scaled UP."""
    plan = compress_plan(100 * MIB, 100.0, 1080, threshold_mb=50)
    assert [r["height"] for r in plan] == [720, 480, 360]
    assert [r["crf"] for r in plan] == [23, 25, 28]
    assert all(r["maxrate_kbps"] is None for r in plan)

    short = compress_plan(100 * MIB, 100.0, 360, threshold_mb=50)
    assert all(r["height"] is None for r in short)

    # An unknown source height is assumed worth scaling.
    unknown = compress_plan(100 * MIB, 100.0, None, threshold_mb=50)
    assert all(r["height"] in (720, 480, 360) for r in unknown)


def test_plan_caps_the_bitrate_when_the_source_is_rate_starved():
    """A slow, small source must never be encoded LARGER than itself.

    600 s at 2.5 Mbps would predict 187 MB from a 100 MB original, so that
    CRF rung is capped at the original's bitrate-equivalent (audio paid off
    the top); a faster source like 100 s keeps plain CRF everywhere.
    """
    size = 100 * MIB
    plan = compress_plan(size, 600.0, None, threshold_mb=50)
    assert len(plan) == 3
    assert plan[0]["maxrate_kbps"] is not None
    # The cap keeps the whole rung's output at or under the original.
    assert plan[0]["maxrate_kbps"] * 1000 * 600 / 8 < size
    assert plan[0]["maxrate_kbps"] >= 64  # never squeezed out of existence
    # 480p (1.2 Mbps) and 360p (0.7 Mbps) predict below the original here.
    assert plan[1]["maxrate_kbps"] is None
    assert plan[2]["maxrate_kbps"] is None

    quick = compress_plan(100 * MIB, 100.0, None, threshold_mb=50)
    assert all(r["maxrate_kbps"] is None for r in quick)


def test_plan_has_no_rung_when_audio_alone_overflows():
    """52 MB over an hour is ~120 kbps total: the audio track eats it all."""
    assert compress_plan(52 * MIB, 3600.0, None, threshold_mb=50) == []


# ------------------------------------------------------------------ service


def test_compress_video_builds_the_ladder_argv(monkeypatch, tmp_path):
    """The ffmpeg argv must scale to the rung's height and carry its CRF."""
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        Path(cmd[-1]).write_bytes(b"compressed-bytes")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(asr_service.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(asr_service.subprocess, "run", fake_run)

    src = tmp_path / "media.mp4"
    src.write_bytes(b"source")
    out = tmp_path / "out" / "compressed.mp4"
    result = asr_service.compress_video(src, out, height=480, crf=25)

    assert result == out
    assert out.read_bytes() == b"compressed-bytes"
    cmd = captured["cmd"]
    assert "scale=-2:480" in cmd
    assert cmd[cmd.index("-crf") + 1] == "25"
    assert cmd[cmd.index("-vf") + 1] == "scale=-2:480"
    assert "libx264" in cmd
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert str(src) in cmd


def test_compress_video_keeps_source_resolution_and_can_cap_bitrate(monkeypatch, tmp_path):
    """height=None omits the -vf scale; a cap adds maxrate/bufsize, not a target."""
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        Path(cmd[-1]).write_bytes(b"compressed-bytes")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(asr_service.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(asr_service.subprocess, "run", fake_run)

    src = tmp_path / "media.mp4"
    src.write_bytes(b"source")
    out = tmp_path / "out.mp4"
    asr_service.compress_video(src, out, height=None, crf=23, maxrate_kbps=900)

    cmd = captured["cmd"]
    assert not any(arg.startswith("scale=") for arg in cmd)
    assert "-vf" not in cmd
    assert cmd[cmd.index("-maxrate") + 1] == "900k"
    assert cmd[cmd.index("-bufsize") + 1] == "1800k"
    assert cmd[cmd.index("-crf") + 1] == "23"


def test_probe_media_video_tolerates_garbage(tmp_path):
    """A non-media file probes to (None, None) instead of raising."""
    junk = tmp_path / "media.mp4"
    junk.write_bytes(b"not a video at all")
    assert asr_service.probe_media_video(junk) == (None, None)


# ------------------------------------------------------------------ handler


def test_compress_skips_when_the_original_is_small(monkeypatch):
    """At or under the threshold the job succeeds, notes why, writes nothing."""
    video_id = _video_with_media()
    calls: list[dict] = []
    monkeypatch.setattr(
        asr_service,
        "compress_video",
        _recording_compress({(720, 23): 1}, calls),
    )

    job = _run_compress_job(video_id)

    assert job.status == "succeeded", job.last_error
    assert calls == []
    assert not storage.compressed_media_file(video_id).exists()
    note = (job.payload or {}).get("compress") or {}
    assert "50 MB delivery threshold" in note.get("skipped_reason", "")


def test_compress_picks_the_first_rung_under_target(monkeypatch):
    """720p misses, 480p fits: stop there and promote `compressed.mp4`."""
    video_id = _video_with_media()
    monkeypatch.setattr(settings, "compress_threshold_mb", 0)
    monkeypatch.setattr(settings, "compress_target_mb", 1)  # 1 MiB
    sizes = {(720, 23): 2 * 1024 * 1024, (480, 25): 512 * 1024, (360, 28): 64}
    calls: list[dict] = []
    monkeypatch.setattr(
        asr_service, "compress_video", _recording_compress(sizes, calls)
    )

    job = _run_compress_job(video_id)

    assert job.status == "succeeded", job.last_error
    assert [(c["height"], c["crf"]) for c in calls] == [(720, 23), (480, 25)]
    dest = storage.compressed_media_file(video_id)
    assert dest.exists()
    assert dest.stat().st_size == 512 * 1024
    assert storage.find_compressed_file(video_id) == dest
    # No rung temp files survive the promotion.
    assert [p.name for p in dest.parent.iterdir() if p.name.startswith("compressed-")] == []
    # The artifact size is recorded on the job row's metadata.
    note = (job.payload or {}).get("compress") or {}
    assert note.get("size_bytes") == 512 * 1024
    assert note.get("height") == 480
    assert note.get("crf") == 25


def test_compress_fails_when_the_ladder_never_fits(monkeypatch):
    """Nothing under target: the job fails permanently (bot -> portal link)."""
    video_id = _video_with_media()
    monkeypatch.setattr(settings, "compress_threshold_mb", 0)
    monkeypatch.setattr(settings, "compress_target_mb", 0)
    sizes = {(720, 23): 3000, (480, 25): 2000, (360, 28): 1000}
    calls: list[dict] = []
    monkeypatch.setattr(
        asr_service, "compress_video", _recording_compress(sizes, calls)
    )

    job = _run_compress_job(video_id)

    assert job.status == "failed", job.last_error
    assert [(c["height"], c["crf"]) for c in calls] == [(720, 23), (480, 25), (360, 28)]
    # No artifact, no rung temp files left behind.
    assert not storage.compressed_media_file(video_id).exists()
    assert [p for p in (storage.video_dir(video_id)).iterdir() if p.name.startswith("compressed-")] == []
    assert "delivery ladder is exhausted" in (job.last_error or "")


def test_compress_passes_the_delivery_timeout(monkeypatch):
    """Each rung's ffmpeg call carries `COMPRESSION_TIMEOUT_SECONDS`."""
    video_id = _video_with_media()
    monkeypatch.setattr(settings, "compress_threshold_mb", 0)
    monkeypatch.setattr(settings, "compress_target_mb", 50)
    monkeypatch.setattr(settings, "compression_timeout_seconds", 77)
    sizes = {(720, 23): 4}
    calls: list[dict] = []
    monkeypatch.setattr(
        asr_service, "compress_video", _recording_compress(sizes, calls)
    )

    job = _run_compress_job(video_id)

    assert job.status == "succeeded", job.last_error
    assert calls and calls[0]["timeout_seconds"] == 77


def test_compressed_output_is_idempotent(monkeypatch):
    """A second run must not re-encode an existing compressed.mp4."""
    video_id = _video_with_media()
    dest = storage.compressed_media_file(video_id)
    dest.write_bytes(b"already-there")
    monkeypatch.setattr(settings, "compress_threshold_mb", 0)  # pass the size gate
    calls: list[dict] = []
    monkeypatch.setattr(
        asr_service, "compress_video", _recording_compress({(720, 23): 1}, calls)
    )

    job = _run_compress_job(video_id)

    assert job.status == "succeeded", job.last_error
    assert calls == []
    assert dest.read_bytes() == b"already-there"
    note = (job.payload or {}).get("compress") or {}
    assert note.get("reused") is True
    assert note.get("size_bytes") == len(b"already-there")


# ------------------------------------------------------- ASR invariant gate


def test_compression_never_feeds_asr(monkeypatch):
    """The plan's gate: ASR always reads the ORIGINAL media, never the copy."""
    video_id = _video_with_media(size=32)
    monkeypatch.setattr(settings, "compress_threshold_mb", 0)
    monkeypatch.setattr(settings, "compress_target_mb", 1)  # any rung output fits
    monkeypatch.setattr(
        asr_service,
        "compress_video",
        _recording_compress({(720, 23): 64, (480, 25): 64, (360, 28): 64}, []),
    )
    assert _run_compress_job(video_id).status == "succeeded"
    compressed = storage.compressed_media_file(video_id)
    assert compressed.exists()

    seen: dict[str, Path] = {}

    def fake_extract(media_path, out_path, timeout_seconds=None):
        seen["media"] = Path(media_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RIFF" + b"\x00" * 1000)
        return out_path

    monkeypatch.setattr(asr_service, "extract_audio", fake_extract)
    monkeypatch.setattr(asr_service, "extract_mp3", lambda *a, **k: None)

    with SessionLocal() as db:
        job = queue.enqueue(
            db, "extract_audio", video_id=video_id, payload={"chain": []}
        )
        db.commit()
        run_job(db, job)
        assert job.status == "succeeded", job.last_error

    assert seen["media"] == storage.find_media_file(video_id)
    assert seen["media"].name == "media.mp4"
    assert seen["media"] != compressed
    assert "compressed" not in str(seen["media"])


def test_transcribe_reads_the_original_even_with_a_copy(monkeypatch):
    """Focused gate for the transcribe handler: its input path is unchanged."""
    video_id = _video_with_media(size=32)
    storage.compressed_media_file(video_id).write_bytes(b"delivery-copy")

    seen: dict[str, Path] = {}

    def fake_extract(media_path, out_path, timeout_seconds=None):
        seen["media"] = Path(media_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RIFF" + b"\x00" * 1000)
        return out_path

    monkeypatch.setattr(asr_service, "extract_audio", fake_extract)
    monkeypatch.setattr(asr_service, "extract_mp3", lambda *a, **k: None)
    monkeypatch.setattr(asr_service, "transcribe_audio", lambda audio: [])

    with SessionLocal() as db:
        job = queue.enqueue(
            db, "transcribe", video_id=video_id, payload={"chain": []}
        )
        db.commit()
        run_job(db, job)
        assert job.status == "succeeded", job.last_error

    assert seen["media"].name == "media.mp4"
    assert "compressed" not in str(seen["media"])


# ------------------------------------------------------------------ routes


def _upload_with_bytes(client, auth_headers) -> tuple[str, bytes]:
    video = upload_video(client, auth_headers)
    media = storage.find_media_file(video["id"])
    assert media is not None
    return video["id"], media.read_bytes()


def test_compressed_variant_serves_the_copy(client, auth_headers):
    video_id, _original = _upload_with_bytes(client, auth_headers)
    payload = b"COMPRESSED" + b"\x01" * 200
    storage.compressed_media_file(video_id).write_bytes(payload)

    resp = client.get(
        f"{API}/videos/{video_id}/media",
        params={"variant": "compressed"},
        headers=auth_headers,
    )

    assert resp.status_code == 200, resp.text
    assert resp.headers["x-compressed"] == "true"
    assert resp.content == payload


def test_compressed_variant_not_ready_answers_one_code(client, auth_headers):
    """A small original has (and needs) no copy: 404 compressed_not_ready, no job."""
    video_id, _original = _upload_with_bytes(client, auth_headers)
    assert storage.find_compressed_file(video_id) is None

    resp = client.get(
        f"{API}/videos/{video_id}/media",
        params={"variant": "compressed"},
        headers=auth_headers,
    )

    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "compressed_not_ready"
    with SessionLocal() as db:
        rows = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "compress_video")
            .all()
        )
        assert rows == []


def test_missing_original_also_answers_compressed_not_ready(client, auth_headers):
    """An ingest that never downloaded yet has no variant to serve, not a 500."""
    with SessionLocal() as db:
        video = Video(
            source_url=f"https://example.com/never-{uuid4().hex}",
            platform="youtube",
            status="pending",
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        video_id = str(video.id)

    resp = client.get(
        f"{API}/videos/{video_id}/media",
        params={"variant": "compressed"},
        headers=auth_headers,
    )

    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "compressed_not_ready"


def test_big_original_enqueues_compress_and_answers_not_ready(client, auth_headers):
    """>50 MB and no copy: 404 + the compress job, queued exactly once."""
    video_id, _original = _upload_with_bytes(client, auth_headers)
    fixture = large_media_file()
    if fixture is None:
        pytest.skip("ffmpeg is unavailable for the large fixture")
    assert fixture.stat().st_size > 50 * MIB
    storage.find_media_file(video_id).write_bytes(fixture.read_bytes())

    first = client.get(
        f"{API}/videos/{video_id}/media",
        params={"variant": "compressed"},
        headers=auth_headers,
    )
    assert first.status_code == 404, first.text
    assert first.json()["error"]["code"] == "compressed_not_ready"

    second = client.get(
        f"{API}/videos/{video_id}/media",
        params={"variant": "compressed"},
        headers=auth_headers,
    )
    assert second.status_code == 404, second.text
    assert second.json()["error"]["code"] == "compressed_not_ready"

    with SessionLocal() as db:
        rows = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "compress_video")
            .all()
        )
        assert len(rows) == 1
        assert rows[0].status == "queued"
        assert rows[0].priority == 90


def test_variant_validation_and_original_unchanged(client, auth_headers):
    video_id, original = _upload_with_bytes(client, auth_headers)

    bogus = client.get(
        f"{API}/videos/{video_id}/media",
        params={"variant": "bogus"},
        headers=auth_headers,
    )
    assert bogus.status_code == 422, bogus.text
    assert bogus.json()["error"]["code"] == "validation_failed"

    default = client.get(f"{API}/videos/{video_id}/media", headers=auth_headers)
    assert default.status_code == 200
    assert default.headers["x-compressed"] == "false"
    assert default.content == original

    explicit = client.get(
        f"{API}/videos/{video_id}/media",
        params={"variant": "original"},
        headers=auth_headers,
    )
    assert explicit.status_code == 200
    assert explicit.content == original


def test_compressed_variant_is_authorized_like_the_original(client, auth_headers):
    """A private folder answers 404 before the variant is even considered."""
    folder = make_folder(client, auth_headers, visibility="private")
    video = upload_video(client, auth_headers, folder_id=folder["id"])
    storage.compressed_media_file(video["id"]).write_bytes(b"secret-copy")

    resp = client.get(
        f"{API}/videos/{video['id']}/media", params={"variant": "compressed"}
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"


def test_public_folder_visitor_reads_both_variants(client, auth_headers):
    """Whoever can read the original can read the compressed copy, and only that."""
    folder = make_folder(client, auth_headers, visibility="public")
    video = upload_video(client, auth_headers, folder_id=folder["id"])
    storage.compressed_media_file(video["id"]).write_bytes(b"visitor-copy")

    original = client.get(f"{API}/videos/{video['id']}/media", headers=guest_headers())
    variant = client.get(
        f"{API}/videos/{video['id']}/media",
        params={"variant": "compressed"},
        headers=guest_headers(),
    )
    assert original.status_code == 200, original.text
    assert variant.status_code == 200, variant.text
    assert variant.headers["x-compressed"] == "true"
    assert variant.content == b"visitor-copy"


def test_compress_route_is_operator_only_and_needs_media(client, auth_headers):
    video_id, _original = _upload_with_bytes(client, auth_headers)

    resp = client.post(f"{API}/videos/{video_id}/compress", headers=auth_headers)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["kind"] == "compress_video"
    assert body["status"] == "queued"
    with SessionLocal() as db:
        row = db.get(Job, body["id"])
        assert row is not None
        assert row.kind == "compress_video"
        assert row.status == "queued"
        # A standalone job: no chain, so nothing else was enqueued by it.
        assert "chain" not in (row.payload or {}) or row.payload == {"chain": []}

    plain = user_headers(ensure_user("plain-compressor"))
    denied = client.post(f"{API}/videos/{video_id}/compress", headers=plain)
    assert denied.status_code == 403, denied.text

    with SessionLocal() as db:
        bare = Video(
            source_url=f"https://example.com/no-media-{uuid4().hex}",
            platform="youtube",
            status="pending",
        )
        db.add(bare)
        db.commit()
        db.refresh(bare)
        bare_id = str(bare.id)
    missing = client.post(f"{API}/videos/{bare_id}/compress", headers=auth_headers)
    assert missing.status_code == 409, missing.text
    assert missing.json()["error"]["code"] == "not_ready"


def test_compress_route_is_idempotent(client, auth_headers):
    """Pressing the operator button twice queues exactly one job."""
    video_id, _original = _upload_with_bytes(client, auth_headers)

    first = client.post(f"{API}/videos/{video_id}/compress", headers=auth_headers)
    assert first.status_code == 202, first.text
    second = client.post(f"{API}/videos/{video_id}/compress", headers=auth_headers)
    assert second.status_code == 202, second.text
    assert second.json()["id"] == first.json()["id"]

    with SessionLocal() as db:
        rows = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.kind == "compress_video")
            .all()
        )
        assert len(rows) == 1


def test_range_request_on_the_compressed_variant(client, auth_headers):
    video_id, _original = _upload_with_bytes(client, auth_headers)
    payload = bytes(range(256))
    storage.compressed_media_file(video_id).write_bytes(payload)

    resp = client.get(
        f"{API}/videos/{video_id}/media",
        params={"variant": "compressed"},
        headers={**auth_headers, "Range": "bytes=10-19"},
    )

    assert resp.status_code == 206, resp.text
    assert resp.headers["content-range"] == f"bytes 10-19/{len(payload)}"
    assert resp.headers["x-compressed"] == "true"
    assert resp.content == payload[10:20]


# ------------------------------------------------------------- integration


def _ffprobe_validated(path: Path) -> dict[str, str]:
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    proc = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "stream=codec_name,width,height",
            "-of", "csv=p=0", str(path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    codecs = [ln.split(",")[0] for ln in proc.stdout.decode().splitlines() if ln.strip()]
    return {"codecs": codecs}


def test_large_original_compresses_under_the_send_limit():
    """Integration: a real >50 MB 720p fixture lands <=50 MB and ffprobe-validates."""
    fixture = large_media_file()
    if fixture is None:
        pytest.skip("ffmpeg is unavailable for the large fixture")
    original_size = fixture.stat().st_size
    assert original_size > 50 * MIB

    video_id = _video_with_media()
    storage.find_media_file(video_id).write_bytes(fixture.read_bytes())

    job = _run_compress_job(video_id)

    assert job.status == "succeeded", job.last_error
    dest = storage.compressed_media_file(video_id)
    assert dest.exists()
    compressed_size = dest.stat().st_size
    assert 0 < compressed_size <= 50 * MIB, compressed_size
    note = (job.payload or {}).get("compress") or {}
    assert note.get("size_bytes") == compressed_size
    # Runs, and carries h264+aac.
    codecs = _ffprobe_validated(dest)["codecs"]
    assert "h264" in codecs
    assert "aac" in codecs