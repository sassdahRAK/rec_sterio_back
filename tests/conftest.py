"""Pytest fixtures.

The tests run against a real PostgreSQL database (JSONB is required by the
data model) named `recsterio_test`. Set TEST_DATABASE_URL to override.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

TEST_DB_NAME = "recsterio_test"
ADMIN_EMAIL = "admin@test.local"
ADMIN_PASSWORD = "test-password"
BOT_TOKEN = "test-bot-service-token"

_ADMIN_DSN = "postgresql://postgres:postgres@localhost:5432/postgres"
_TEST_DSN = (
    os.environ.get("TEST_DATABASE_URL")
    or f"postgresql+psycopg://postgres:postgres@localhost:5432/{TEST_DB_NAME}"
)


def _ensure_test_database() -> None:
    import psycopg

    try:
        with psycopg.connect(_ADMIN_DSN, autocommit=True) as conn:
            row = conn.execute(
                "select 1 from pg_database where datname = %s", (TEST_DB_NAME,)
            ).fetchone()
            if row is None:
                conn.execute(f'create database "{TEST_DB_NAME}"')
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PostgreSQL is required for the test suite. Start it with "
            "`brew services start postgresql@15` and make sure the postgres role "
            f"can log in with the password 'postgres'. Original error: {exc}"
        ) from exc


_ensure_test_database()

_MEDIA_ROOT = tempfile.mkdtemp(prefix="recsterio-test-media-")
_COOKIES_ROOT = tempfile.mkdtemp(prefix="recsterio-test-cookies-")

# The environment must be configured *before* app.config is imported.
os.environ["DATABASE_URL"] = _TEST_DSN
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["ADMIN_EMAIL"] = ADMIN_EMAIL
os.environ["ADMIN_PASSWORD"] = ADMIN_PASSWORD
os.environ["BOT_API_TOKEN"] = BOT_TOKEN
os.environ["MEDIA_ROOT"] = _MEDIA_ROOT
os.environ["COOKIES_ROOT"] = _COOKIES_ROOT
os.environ["WORKER_AUTOSTART"] = "false"
os.environ["WORKER_POLL_SECONDS"] = "0.2"
os.environ["GEMINI_API_KEY"] = ""
os.environ["LOGIN_RATE_LIMIT"] = "10000"
os.environ["VIDEO_CREATE_RATE_LIMIT"] = "10000"
# The developer's real .env (loaded by pydantic-settings) must not leak into
# the suite: a configured bot username changes link-code responses, the same
# way a real ADMIN_PASSWORD would change every login test.
os.environ["TELEGRAM_BOT_USERNAME"] = ""

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.auth import seed_admin_user  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_admin_user(db)
    with TestClient(app) as test_client:
        yield test_client
    shutil.rmtree(_MEDIA_ROOT, ignore_errors=True)
    shutil.rmtree(_COOKIES_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_database(client):
    """Wipe transactional data between tests (admin user is kept).

    Order matters: children before parents. Folders cascade to members, shares,
    passcodes and comments, but videos point at folders, so they go first.
    """
    with engine.begin() as conn:
        # Accounts a test created go first: they cascade into folders,
        # memberships, comments and passcode grants. The seeded bootstrap admin
        # is preserved, and any other account is removed so a test that invites
        # "user@test.local" cannot collide with a previous run.
        conn.execute(
            text("delete from admin_users where email <> :keep"), {"keep": ADMIN_EMAIL}
        )
        # The seeded admin survives the wipe above, so its Telegram link (and
        # any outstanding link code) would too: reset both, or a test that
        # linked the operator poisons the next one that asserts "unlinked".
        conn.execute(text("update admin_users set telegram_user_id = null"))
        for table in (
            "link_codes",
            "passcode_grants",
            "folder_passcodes",
            "share_links",
            "folder_comments",
            "folder_members",
            "merges",
            "jobs",
            "videos",
            "transcripts",
            "folders",
            "bot_users",
        ):
            conn.execute(text(f"delete from {table}"))
    # The per-user cookies files live on disk, outside the database wipe.
    for name in os.listdir(_COOKIES_ROOT):
        os.unlink(os.path.join(_COOKIES_ROOT, name))
    yield


@pytest.fixture(scope="session")
def auth_headers(client):
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


@pytest.fixture
def bot_headers():
    return {"Authorization": f"Bearer {BOT_TOKEN}"}
