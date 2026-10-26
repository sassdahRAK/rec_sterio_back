"""Referential integrity for constraints added after the first release.

`create_all` builds a correct schema on a fresh database, so a missing foreign
key is invisible to the rest of the suite: it only shows up on a database that
predates the constraint. These tests therefore drop the constraint first to
reproduce that state, then assert `init_db()` repairs it.

Regression test for: deleting a folder left its videos with a `folder_id`
pointing at a folder that no longer existed, so they vanished from folder
listings and ZIP exports while still appearing in the global video list.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db import _add_missing_constraints, engine
from tests.helpers import make_folder, upload_video

API = "/api/v1"
CONSTRAINT = "videos_folder_id_fkey"


@pytest.fixture(autouse=True)
def restore_constraint():
    """Never leave the schema altered if a test fails mid-way.

    One test drops the constraint to reproduce the pre-upgrade state, so a
    failure between the drop and the repair would otherwise leak a schema
    change into every later test.
    """
    yield
    _add_missing_constraints()


def _drop_constraint() -> None:
    """Put the database in the pre-upgrade state: the column, no constraint."""
    with engine.begin() as conn:
        conn.execute(text(f"alter table videos drop constraint if exists {CONSTRAINT}"))


def _constraint_exists() -> bool:
    with engine.connect() as conn:
        return (
            conn.execute(
                text(
                    "select 1 from pg_constraint c"
                    " where c.conname = :name and c.conrelid = 'videos'::regclass"
                ),
                {"name": CONSTRAINT},
            ).fetchone()
            is not None
        )


def _dangling_count() -> int:
    with engine.connect() as conn:
        return conn.execute(
            text(
                "select count(*) from videos v"
                " where v.folder_id is not null"
                " and not exists (select 1 from folders f where f.id = v.folder_id)"
            )
        ).scalar_one()


def test_deleting_a_folder_detaches_its_videos(client, auth_headers):
    """The behaviour the comment in folders.py promises."""
    folder = make_folder(client, auth_headers, name="Detach on delete")
    upload_video(client, auth_headers, folder_id=folder["id"], title="Orphan me")

    resp = client.delete(f"{API}/folders/{folder['id']}", headers=auth_headers)
    assert resp.status_code == 204, resp.text

    with engine.connect() as conn:
        folder_id = conn.execute(
            text("select folder_id from videos order by created_at desc limit 1")
        ).scalar_one()
    assert folder_id is None


def test_the_constraint_exists(client):
    assert _constraint_exists()


def test_orphaned_rows_are_repaired_and_the_constraint_added(client, auth_headers):
    """A pre-upgrade database is upgraded in place, without losing media."""
    assert _constraint_exists()
    _drop_constraint()
    assert not _constraint_exists()

    folder = make_folder(client, auth_headers, name="Pre-upgrade folder")
    video = upload_video(client, auth_headers, folder_id=folder["id"], title="Pre-upgrade")

    # Delete while the constraint is absent: this is the bug.
    resp = client.delete(f"{API}/folders/{folder['id']}", headers=auth_headers)
    assert resp.status_code == 204, resp.text
    assert _dangling_count() == 1, "expected the orphan the bug produces"

    _add_missing_constraints()

    assert _constraint_exists()
    assert _dangling_count() == 0

    # The video survived: detaching is not deleting.
    with engine.connect() as conn:
        title = conn.execute(
            text("select title from videos where id = :id"), {"id": video["id"]}
        ).scalar_one()
    assert title == "Pre-upgrade"


def test_the_migration_is_idempotent(client):
    _add_missing_constraints()
    _add_missing_constraints()
    assert _constraint_exists()
    assert _dangling_count() == 0
