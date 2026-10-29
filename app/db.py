"""SQLAlchemy engine/session plumbing."""
from __future__ import annotations

from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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
    """Create tables if they do not exist (bootstrap; alembic optional later)."""
    from app import models  # noqa: F401  (register mappers)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    _add_missing_constraints()


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
)


def _add_missing_columns() -> None:
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
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
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
