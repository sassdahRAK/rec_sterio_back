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
)


def _add_missing_columns() -> None:
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        for statement in _ADDED_COLUMNS:
            conn.exec_driver_sql(statement)
