"""SQLAlchemy engine/session plumbing."""
from __future__ import annotations

from datetime import datetime
from typing import Iterator

from sqlalchemy import DateTime, Integer, create_engine, func
from sqlalchemy.engine import Connection
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class SchemaMigration(Base):
    """One row per applied schema version (the migration version stamp).

    The hand-rolled upgrade batch below is far too heavy to re-run on every
    container start: it takes table locks, runs DDL/DML, and two containers
    booting together would fight over it. This table records the version the
    database has already reached, so the batch runs exactly once per database
    and a concurrent boot waits on an advisory lock instead of racing.

    Declared on `Base` (not in `models.py`) on purpose: `create_all` creates it
    on a fresh database, and a test that drops/recreates the schema drops the
    stamp with it — a stale stamp must never outlive the schema it describes.
    """

    __tablename__ = "schema_migrations"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# Bump this whenever `_ADDED_COLUMNS` or `_ADDED_FOREIGN_KEYS` gains a step.
# A database already stamped at this version skips the whole batch.
SCHEMA_VERSION = 1

# Advisory-lock key for the migration. Follows the codebase's
# `hashtextextended(<key>, 0)` pattern (see `app/routers/videos.py`); the xact
# form releases itself when the transaction ends, so a crashed boot cannot
# leave the lock held.
_MIGRATION_LOCK_KEY = "recsterio.schema_migrations"


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they do not exist (bootstrap; alembic optional later).

    `create_all` is cheap and idempotent, so it still runs on every boot: a new
    table (or the stamp table itself) is created even when the version is
    already current. Only the expensive ALTER/UPDATE/DML batch is gated by
    `SCHEMA_VERSION`.
    """
    from app import models  # noqa: F401  (register mappers)

    if engine.dialect.name != "postgresql":
        Base.metadata.create_all(bind=engine)
        return

    # One transaction: lock, create tables, read the stamp, migrate, stamp.
    # A second container booting at the same time blocks on the advisory lock
    # and then sees the bumped version, so the batch never runs twice.
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "select pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (_MIGRATION_LOCK_KEY,),
        )
        Base.metadata.create_all(bind=conn)

        current = _current_schema_version(conn)
        if current >= SCHEMA_VERSION:
            return
        if current == 0 and _migration_already_applied(conn):
            # A database from before the stamp existed: the batch has already
            # been applied by the previous code, so adopt it without replaying
            # any destructive work — just record where it stands.
            _record_schema_version(conn, SCHEMA_VERSION)
            return

        _run_migration_statements(conn)
        _record_schema_version(conn, SCHEMA_VERSION)


def _current_schema_version(conn: Connection) -> int:
    return conn.exec_driver_sql(
        "select coalesce(max(version), 0) from schema_migrations"
    ).scalar_one()


def _record_schema_version(conn: Connection, version: int) -> None:
    conn.exec_driver_sql(
        "insert into schema_migrations (version) values (%s)"
        " on conflict (version) do nothing",
        (version,),
    )


# Proof that the whole batch has run at least once on this database. The
# composite unique index is the last statement in `_ADDED_COLUMNS` and is the
# only guard rail `create_all` can never produce (the model does not declare
# it), so its presence means "already migrated" even when the stamp is empty.
_MIGRATION_MARKER_INDEX = "ix_drive_links_user_provider"


def _migration_already_applied(conn: Connection) -> bool:
    row = conn.exec_driver_sql(
        "select 1 from pg_indexes"
        " where schemaname = current_schema() and indexname = %s",
        (_MIGRATION_MARKER_INDEX,),
    ).fetchone()
    return row is not None


def _run_migration_statements(conn: Connection) -> None:
    """The whole upgrade batch, in order, inside the caller's transaction."""
    for statement in _ADDED_COLUMNS:
        conn.exec_driver_sql(statement)
    _apply_missing_constraints(conn)


# Columns added after the first release. `create_all` never alters an existing
# table, so a database created before these features existed would otherwise
# keep failing on every insert. Adding them here keeps upgrades to "restart the
# server" instead of "write and run a migration".
_ADDED_COLUMNS = (
    "alter table videos add column if not exists folder_id uuid",
    "alter table videos add column if not exists position integer not null default 0",
    "alter table transcripts add column if not exists cleaned boolean not null default false",
    "alter table admin_users add column if not exists name varchar(255)",
    "alter table admin_users add column if not exists avatar_url text",
    "alter table admin_users add column if not exists google_sub varchar(255)",
    "alter table admin_users add column if not exists role varchar(32) not null default 'admin'",
    # Google users have no password; the column predates them and was NOT NULL.
    "alter table admin_users alter column password_hash drop not null",
    "create unique index if not exists ix_admin_users_google_sub on admin_users (google_sub)",
    # A share link opens a folder for a visitor with no account, so a comment
    # cannot always point at an admin_users row.
    "alter table folder_comments add column if not exists author_label varchar(128) "
    "not null default 'unknown'",
    "alter table folder_comments alter column author_id drop not null",
    # Declared as `index=True` on the model. `create_all` never adds an index to
    # a table that already exists, so a database created before this existed
    # would seq-scan every actor-filtered listing (`list_videos`,
    # `list_dataset_cards`, the dedup paths).
    "create index if not exists ix_videos_created_by on videos (created_by)",
    # A 2-hour YouTube video is ~7 GB, which overflows a 32-bit integer and
    # fails the post-download UPDATE outright (leaving the job orphaned at
    # 'running'). `create_all` never widens an existing column, so alter it.
    "alter table videos alter column file_size type bigint",
    # Telegram mirror (instruction.txt #4) and the TvNa collection schema (#6).
    "alter table videos add column if not exists telegram_file_id varchar(255)",
    "alter table videos add column if not exists telegram_message_id bigint",
    "alter table videos add column if not exists tvna_serial integer",
    "alter table videos add column if not exists source_type varchar(32)",
    "alter table videos add column if not exists author_handle varchar(255)",
    "alter table videos add column if not exists date_posted varchar(16)",
    "alter table videos add column if not exists caption text",
    "alter table videos add column if not exists location_tag text",
    "alter table videos add column if not exists collection_method varchar(32)",
    "alter table videos add column if not exists collected_by varchar(128)",
    "alter table videos add column if not exists collection_date varchar(16)",
    # Postgres treats NULLs as distinct in a unique index, so unfiled videos
    # (tvna_serial IS NULL) do not collide.
    "create unique index if not exists ix_videos_tvna_serial on videos (tvna_serial)",
    # A fork points at the row it was copied from. Self-referencing, SET NULL:
    # deleting the original must leave the copy standing — that is the point of
    # a copy. Also indexed because the de-dupe lookups filter on it.
    "alter table videos add column if not exists copied_from_id uuid",
    "create index if not exists ix_videos_copied_from_id on videos (copied_from_id)",
    # Telegram account linking (console <-> bot). One Telegram account may be
    # bound to at most one console account, enforced by a unique index so a
    # race in the link flow cannot corrupt the pairing.
    "alter table admin_users add column if not exists telegram_user_id bigint",
    "create unique index if not exists ix_admin_users_telegram_user_id on admin_users (telegram_user_id)",
    # Role-system rename: the account role 'member' became 'user', and the
    # stored actor strings had to follow, because access control compares
    # them by string equality (`video_role`, `access.py`) — without the
    # rename, every video a member ingested would fall out of its creator's
    # reach. One-time, but written to be idempotent so every boot can run
    # them: after the first pass no 'member' prefix or role remains for the
    # WHERE clauses to match. (The LIKE patterns escape '%' as '%%' because
    # psycopg parses the statement for placeholders.)
    "update admin_users set role = 'user' where role = 'member'",
    "update videos set created_by = 'user:' || split_part(created_by, ':', 2) where created_by like 'member:%%'",
    "update videos set collected_by = 'user:' || split_part(collected_by, ':', 2) where collected_by like 'member:%%'",
    "update transcripts set created_by = 'user:' || split_part(created_by, ':', 2) where created_by like 'member:%%'",
    "update share_links set created_by = 'user:' || split_part(created_by, ':', 2) where created_by like 'member:%%'",
    "update folder_passcodes set created_by = 'user:' || split_part(created_by, ':', 2) where created_by like 'member:%%'",
    "update folder_comments set author_label = 'user:' || split_part(author_label, ':', 2) where author_label like 'member:%%'",
    # Personal-cloud storage: a video's bytes may live in the contributor's own
    # Google Drive / OneDrive. `local_path` stays NULL for these rows, so the
    # media/audio/ZIP readers fall through to the cloud link (and report the
    # unit "unavailable" when the link is gone, never a 500).
    "alter table videos add column if not exists storage_backend varchar(16)",
    "alter table videos add column if not exists drive_file_id varchar(255)",
    "alter table videos add column if not exists drive_link_id uuid",
    "create index if not exists ix_videos_drive_link_id on videos (drive_link_id)",
    # One live drive link per (account, provider): enforced at the DB level so
    # two concurrent connect requests cannot create duplicate rows.
    "create unique index if not exists ix_drive_links_user_provider on drive_links (user_id, provider)",
)


def _add_missing_columns() -> None:
    """Standalone entry point (manual repairs); the boot path runs the same
    statements, plus the constraints, inside the gated migration transaction."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        for statement in _ADDED_COLUMNS:
            conn.exec_driver_sql(statement)


# Foreign keys added after the first release. `create_all` only creates a table
# once, so a constraint declared on a column that predates it is never created
# -- and the ORM's `ondelete` is then a no-op. The damage is silent: deleting a
# folder leaves its videos pointing at a folder that no longer exists, so they
# drop out of every folder listing and out of ZIP exports while still showing up
# in the global video list.
#
# (table, constraint name, column, parent table, parent column, on delete)
_ADDED_FOREIGN_KEYS = (
    ("videos", "videos_folder_id_fkey", "folder_id", "folders", "id", "set null"),
    ("videos", "videos_copied_from_id_fkey", "copied_from_id", "videos", "id", "set null"),
    # A cloud-stored video points at the drive link its bytes live under;
    # deleting the link leaves the row intact but storage-less (revoked).
    ("videos", "videos_drive_link_id_fkey", "drive_link_id", "drive_links", "id", "set null"),
)


def _add_missing_constraints() -> None:
    """Standalone entry point (tests and manual repairs); the boot path runs
    `_apply_missing_constraints` inside the gated migration transaction."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        _apply_missing_constraints(conn)


def _apply_missing_constraints(conn: Connection) -> None:
    for table, name, column, parent, parent_column, on_delete in _ADDED_FOREIGN_KEYS:
        # Orphaned rows would make ADD CONSTRAINT fail, so repair them first.
        if parent == table:
            # Self-reference (videos.copied_from_id -> videos.id): the
            # subquery's range var shadows the outer one, so the unaliased
            # form below would compare the inner row to itself and null out
            # every value. Alias both sides explicitly.
            conn.exec_driver_sql(
                f"update {table} as t set {column} = null"
                f" where t.{column} is not null"
                f" and not exists (select 1 from {parent} as p"
                f" where p.{parent_column} = t.{column})"
            )
        else:
            conn.exec_driver_sql(
                f"update {table} set {column} = null"
                f" where {column} is not null"
                f" and not exists (select 1 from {parent}"
                f" where {parent}.{parent_column} = {table}.{column})"
            )
        exists = conn.exec_driver_sql(
            "select 1 from pg_constraint c"
            " where c.conname = %s and c.conrelid = %s::regclass",
            (name, table),
        ).fetchone()
        if exists is None:
            conn.exec_driver_sql(
                f"alter table {table} add constraint {name}"
                f" foreign key ({column}) references {parent}({parent_column})"
                f" on delete {on_delete}"
            )
