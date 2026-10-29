"""F9: the schema upgrade batch is version-stamped and runs once per database.

`init_db()` used to replay the whole ALTER/UPDATE batch on every process start,
so every API/worker restart re-ran the DDL/DML and concurrent containers
fought over the same locks. It now records the applied version in
`schema_migrations` (under a Postgres advisory lock) and skips the batch when
the database is already at `SCHEMA_VERSION`.

Two paths are pinned here:

* an unstamped database below target runs the batch once, and the next boot
  issues no ALTER/UPDATE at all;
* a database that was already migrated before the stamp existed is adopted —
  stamped without replaying the batch.
"""
from __future__ import annotations

from typing import Iterator

import pytest
from sqlalchemy import event, text

from app import db as db_module
from app.db import SCHEMA_VERSION, engine, init_db

# Every statement the batch can emit starts with one of these (see
# `_ADDED_COLUMNS` / `_add_missing_constraints`); `create_all` never does.
_MIGRATION_PREFIXES = ("alter", "update")


def _capture_statements() -> tuple[list[str], object]:
    """Record every SQL statement the engine executes, in order."""
    statements: list[str] = []

    def listener(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", listener)
    return statements, listener


def _migration_statements(statements: list[str]) -> list[str]:
    return [
        statement
        for statement in statements
        if statement.lstrip().lower().startswith(_MIGRATION_PREFIXES)
    ]


def _stamped_versions() -> list[int]:
    with engine.connect() as conn:
        return list(
            conn.execute(text("select version from schema_migrations order by version")).scalars()
        )


@pytest.fixture
def unstamped() -> Iterator[None]:
    """Forget the version stamp, then leave a current one behind for later tests."""
    with engine.begin() as conn:
        conn.execute(text("delete from schema_migrations"))
    yield
    init_db()
    assert _stamped_versions() == [SCHEMA_VERSION]


def _init_db_capturing() -> list[str]:
    statements, listener = _capture_statements()
    try:
        init_db()
    finally:
        event.remove(engine, "before_cursor_execute", listener)
    return statements


def test_second_boot_runs_no_alter_or_update_ddl(client, unstamped, monkeypatch):
    """The batch applies once; the next process start is a pure no-op."""
    # Pretend the marker is absent so the first call cannot take the adoption
    # shortcut — this is the "database is genuinely below target" path.
    real_marker = db_module._migration_already_applied
    monkeypatch.setattr(db_module, "_migration_already_applied", lambda conn: False)

    first = _init_db_capturing()

    assert _stamped_versions() == [SCHEMA_VERSION]
    assert _migration_statements(first), "the first boot must apply the batch"
    assert any(
        "pg_advisory_xact_lock" in statement for statement in first
    ), "the migration must be serialised by the advisory lock"

    # Restore the real marker check: the stamp alone must now be enough.
    monkeypatch.setattr(db_module, "_migration_already_applied", real_marker)

    second = _init_db_capturing()

    assert _migration_statements(second) == [], (
        "a boot at the target version must not replay ALTER/UPDATE DDL"
    )
    assert _stamped_versions() == [SCHEMA_VERSION], "the stamp is not duplicated"


def test_existing_database_is_adopted_without_replaying_the_batch(client, unstamped):
    """A live database migrated by the pre-stamp code boots without redoing work.

    Its stamp table is empty but the schema is fully migrated (the marker index
    from the last batch statement exists), which is exactly the state of a
    deployment on its first boot after this change.
    """
    statements = _init_db_capturing()

    assert _migration_statements(statements) == [], (
        "adoption must stamp the version, not re-run destructive work"
    )
    assert _stamped_versions() == [SCHEMA_VERSION]
