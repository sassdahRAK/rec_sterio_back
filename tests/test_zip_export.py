"""Dataset export: the archive is what leaves the building.

The ZIP layout is part of the dataset contract (#6) — a downstream user unpacks
it and expects `<folder>/<unit>/{video,audio,transcript}`. These tests pin the
layout, the Khmer/English language tags, and the guard rails that stop a
pathological tree from producing an unbounded archive.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.db import SessionLocal
from app.models import AdminUser, Folder, Transcript, Video
from app.services import storage, zip_export


def _admin_id() -> object:
    with SessionLocal() as db:
        return db.execute(
            select(AdminUser).where(AdminUser.email == "admin@test.local")
        ).scalar_one().id


def _folder(name, parent_id=None, visibility="private"):
    with SessionLocal() as db:
        folder = Folder(
            name=name,
            parent_id=parent_id,
            visibility=visibility,
            owner_id=_admin_id(),
        )
        db.add(folder)
        db.commit()
        db.refresh(folder)
        return folder.id


def _video(folder_id=None, title="Clip", with_media=False, with_audio=False):
    with SessionLocal() as db:
        video = Video(
            source_url="upload://zip-test.mp4",
            platform="direct",
            title=title,
            folder_id=folder_id,
            created_by="super_admin:admin@test.local",
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        vid = video.id

    if with_media:
        storage.media_file(vid, "mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 64)
    if with_audio:
        storage.audio_file(vid).write_bytes(b"RIFF" + b"\x00" * 32)
    return vid


def _transcript(video_id, content="", segments=None, status="ready", version=1):
    with SessionLocal() as db:
        row = Transcript(
            video_id=video_id,
            version=version,
            source="manual",
            content=content,
            segments=segments,
            status=status,
        )
        db.add(row)
        db.commit()
        return row.id


# ------------------------------------------------------------------ naming


def test_safe_name_strips_characters_that_are_unsafe_on_some_filesystems():
    assert zip_export.safe_name('a/b\\c:d*e?f"g<h>i|j', "fallback") == "a_b_c_d_e_f_g_h_i_j"


def test_safe_name_falls_back_when_nothing_usable_remains():
    assert zip_export.safe_name(None, "fallback") == "fallback"
    assert zip_export.safe_name("   ", "fallback") == "fallback"


def test_safe_name_replaces_unsafe_characters_with_underscores():
    assert zip_export.safe_name("///", "fallback") == "___"


def test_safe_name_truncates_a_pathological_title():
    assert len(zip_export.safe_name("x" * 500, "fallback")) == 120


def test_video_slug_carries_the_title_and_the_id_prefix():
    """Readable enough for a human, unique enough to never collide."""
    vid = _video(title="My Clip: Phnom Penh")
    with SessionLocal() as db:
        slug = zip_export.video_slug(db.get(Video, vid))
    assert str(vid)[:8] in slug
    assert "My Clip" in slug


def test_two_videos_with_the_same_title_get_distinct_slugs():
    with SessionLocal() as db:
        a = Video(source_url="upload://a.mp4", platform="direct", title="Dup")
        b = Video(source_url="upload://b.mp4", platform="direct", title="Dup")
        db.add_all([a, b])
        db.commit()
        db.refresh(a)
        db.refresh(b)
        assert zip_export.video_slug(a) != zip_export.video_slug(b)


# --------------------------------------------------------------- transcripts


def test_current_transcript_prefers_a_ready_version_over_a_newer_pending_one():
    vid = _video()
    _transcript(vid, content="raw", version=1, status="ready")
    _transcript(vid, content="still processing", version=2, status="pending")
    with SessionLocal() as db:
        assert zip_export.current_transcript(db, vid).content == "raw"


def test_current_transcript_falls_back_to_the_newest_when_none_are_ready():
    vid = _video()
    _transcript(vid, content="failed run", version=1, status="failed")
    with SessionLocal() as db:
        assert zip_export.current_transcript(db, vid).content == "failed run"


def test_current_transcript_is_none_for_a_video_never_transcribed():
    vid = _video()
    with SessionLocal() as db:
        assert zip_export.current_transcript(db, vid) is None


def test_transcript_text_keeps_the_per_segment_language_tags():
    """The language tags are the point of the dataset — they must survive export."""
    segments = [
        {"language": "km", "text": "សួស្តី"},
        {"language": "en", "text": "hello there"},
    ]
    text = zip_export.transcript_text(type("T", (), {"segments": segments, "content": ""})())
    assert "[km] សួស្តី" in text
    assert "[en] hello there" in text


def test_transcript_text_falls_back_to_the_flat_content_field():
    text = zip_export.transcript_text(
        type("T", (), {"segments": None, "content": "  plain text  "})()
    )
    assert text == "plain text"


def test_transcript_text_of_a_missing_transcript_is_empty():
    assert zip_export.transcript_text(None) == ""


# ------------------------------------------------------------------ archive


def _build(folder_id):
    dest = Path(storage.media_root()) / f"test-export-{uuid4().hex}.zip"
    with SessionLocal() as db:
        count = zip_export.build_folder_zip(db, db.get(Folder, folder_id), dest)
    names = zipfile.ZipFile(dest).namelist()
    dest.unlink(missing_ok=True)
    return count, names


def test_archive_mirrors_the_folder_tree(tmp_path):
    root = _folder("Dataset")
    unit = _video(root, title="Unit one", with_media=True, with_audio=True)
    _transcript(unit, content="hello")
    child = _folder("Sub", parent_id=root)
    _video(child, title="Unit two", with_media=True)

    count, names = _build(root)
    assert count == len(names)
    assert "Dataset/Unit one-%s/video/media.mp4" % str(unit)[:8] in names
    assert "Dataset/Unit one-%s/audio/audio.wav" % str(unit)[:8] in names
    assert "Dataset/Unit one-%s/transcript/Unit one-%s.txt" % (str(unit)[:8], str(unit)[:8]) in names
    # The sub-folder nests under its parent, not at the archive root.
    assert any(n.startswith("Dataset/Sub/") for n in names)
    assert not any(n.startswith("Sub/") for n in names)


def test_a_folder_whose_videos_have_no_media_yet_has_nothing_to_export(tmp_path):
    """A pending video contributes no empty placeholder — only what exists.

    The HTTP layer turns this into a 409 with the folder id, so an operator
    learns *which* folder is not ready rather than downloading an empty file.
    """
    root = _folder("Dataset")
    _video(root, title="No media yet")
    dest = tmp_path / "nothing.zip"
    with SessionLocal() as db:
        with pytest.raises(ValueError, match="no downloadable media"):
            zip_export.build_folder_zip(db, db.get(Folder, root), dest)


def test_an_empty_folder_is_an_error_not_an_empty_archive(tmp_path):
    root = _folder("Nothing here")
    dest = tmp_path / "empty.zip"
    with SessionLocal() as db:
        with pytest.raises(ValueError, match="no downloadable media"):
            zip_export.build_folder_zip(db, db.get(Folder, root), dest)


def test_archive_refuses_to_exceed_the_entry_limit(tmp_path, monkeypatch):
    """A pathological tree must not produce an unbounded archive."""
    root = _folder("Huge")
    for i in range(3):
        _video(root, title=f"clip {i}", with_media=True)

    monkeypatch.setattr(zip_export, "MAX_ENTRIES", 2)
    dest = tmp_path / "capped.zip"
    with SessionLocal() as db:
        with pytest.raises(ValueError, match="entry limit"):
            zip_export.build_folder_zip(db, db.get(Folder, root), dest)
    dest.unlink(missing_ok=True)


def test_archive_refuses_to_exceed_the_size_limit(tmp_path, monkeypatch):
    root = _folder("Heavy")
    _video(root, title="big", with_media=True)

    monkeypatch.setattr(zip_export, "MAX_TOTAL_BYTES", 1)
    dest = tmp_path / "capped.zip"
    with SessionLocal() as db:
        with pytest.raises(ValueError, match="size limit"):
            zip_export.build_folder_zip(db, db.get(Folder, root), dest)
    dest.unlink(missing_ok=True)


def test_exported_transcript_file_contains_the_transcript():
    root = _folder("Dataset")
    unit = _video(root, title="Unit", with_media=True)
    _transcript(unit, content="the transcript body")
    _, names = _build(root)
    transcript_name = next(n for n in names if n.endswith(".txt"))
    with SessionLocal() as db:
        dest = Path(storage.media_root()) / f"check-{uuid4().hex}.zip"
        zip_export.build_folder_zip(db, db.get(Folder, root), dest)
        with zipfile.ZipFile(dest) as zf:
            body = zf.read(transcript_name).decode("utf-8")
        dest.unlink(missing_ok=True)
    assert body == "the transcript body"
