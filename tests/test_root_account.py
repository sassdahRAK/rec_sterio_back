"""The permanent root account: one super admin nobody can lock out.

Root is not a column — it is the `admin_users` row whose email equals the
normalized `settings.ADMIN_EMAIL`, i.e. the identity `seed_admin_user`
creates or promotes on every boot. The rules pinned here:

* privilege edits (role, is_active) targeting the root are refused with
  **403 `root_account_locked`** — by another super admin *and* by the root
  itself (the lock answers before the plain self-guard, different reason);
* the root cannot self-delete either;
* the lock covers privilege fields only — renaming the root or resetting its
  password stays allowed, so the account never becomes unusable;
* the non-root behaviour is untouched: a second super admin still gets the
  422 `self_modification_forbidden` for editing its own privileges and can
  still be demoted or deleted, and plain accounts are fully manageable;
* creating an account with the root's email hits the duplicate guard and
  can never overwrite the root's role.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db import SessionLocal
from app.models import AdminUser
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD

API = "/api/v1"

SECOND_SUPER = "second-super@team.test"
ATTACKER_PASSWORD = "attacker-pass"


def _root_row() -> AdminUser:
    with SessionLocal() as db:
        return db.execute(
            select(AdminUser).where(AdminUser.email == ADMIN_EMAIL)
        ).scalar_one()


def _root_id() -> str:
    return str(_root_row().id)


def _login(client, email: str, password: str) -> dict:
    resp = client.post(f"{API}/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


@pytest.fixture
def attacker(client, auth_headers):
    """A SECOND super_admin — root is unique, so this one is not the root.

    The seeded ADMIN_EMAIL account in the test environment IS the root, so
    the "anyone else" of the permanent-root rule must be minted through the
    API (which only root may call) and must be able to log in.
    """
    resp = client.post(
        f"{API}/users",
        json={"email": SECOND_SUPER, "role": "super_admin", "password": ATTACKER_PASSWORD},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json(), _login(client, SECOND_SUPER, ATTACKER_PASSWORD)


# ------------------------------------------------- the root lock, by another admin


def test_another_super_admin_cannot_demote_the_root(client, auth_headers, attacker):
    second, headers = attacker
    resp = client.patch(
        f"{API}/users/{_root_id()}", json={"role": "user"}, headers=headers
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "root_account_locked"
    assert _root_row().role == "super_admin"
    # The root still runs the console.
    assert _login(client, ADMIN_EMAIL, ADMIN_PASSWORD)


def test_another_super_admin_cannot_deactivate_the_root(client, auth_headers, attacker):
    second, headers = attacker
    resp = client.patch(
        f"{API}/users/{_root_id()}", json={"is_active": False}, headers=headers
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "root_account_locked"
    assert _root_row().is_active is True


def test_the_lock_answers_for_a_combined_privilege_patch(client, auth_headers, attacker):
    """Even role+name in one body is refused entirely — nothing half-applies."""
    second, headers = attacker
    resp = client.patch(
        f"{API}/users/{_root_id()}",
        json={"role": "admin", "name": "Demoted"},
        headers=headers,
    )
    assert resp.status_code == 403, resp.text
    root = _root_row()
    assert root.role == "super_admin"
    assert root.name != "Demoted"


# --------------------------------------------------- the root lock, on the root


def test_the_root_cannot_demote_or_deactivate_itself(client, auth_headers):
    """The root's self-edit gets 403 root_account_locked, not the 422.

    Same refusal as for an attacker: the account is locked because it is
    ROOT, not merely because the caller is the target.
    """
    me_id = _root_id()
    for body in ({"role": "admin"}, {"is_active": False}):
        resp = client.patch(f"{API}/users/{me_id}", json=body, headers=auth_headers)
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "root_account_locked"
    assert _root_row().role == "super_admin"
    assert _root_row().is_active is True


def test_the_root_cannot_delete_itself(client, auth_headers):
    resp = client.request("DELETE", f"{API}/users/me", json={"confirm": True}, headers=auth_headers)
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "root_account_locked"
    # The row is untouched and still signs in.
    assert _login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    assert _root_row().email == ADMIN_EMAIL


# ----------------------------------------- the lock stops at the privilege fields


def test_the_root_lock_does_not_extend_to_cosmetic_fields(client, auth_headers, attacker):
    """Renaming the root stays allowed (by another admin and by itself).

    The lock guards WHO may do what (role, is_active) and deletion — never
    presentation or credentials, so a locked root is still a usable account.
    """
    second, headers = attacker
    renamed = client.patch(
        f"{API}/users/{_root_id()}",
        json={"name": "Root Operator"},
        headers=headers,
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["role"] == "super_admin"
    # Put the seeded row back: the fixture keeps ADMIN_EMAIL between tests
    # but does not reset its columns.
    client.patch(
        f"{API}/users/{_root_id()}", json={"name": None}, headers=auth_headers
    )


# --------------------------------------- non-root behaviour must not regress


def test_a_non_root_super_admin_still_gets_the_422_self_guard(client, attacker):
    """The 422 `self_modification_forbidden` is kept, and kept distinct."""
    second, headers = attacker
    resp = client.patch(f"{API}/users/{second['id']}", json={"role": "user"}, headers=headers)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "self_modification_forbidden"


def test_a_non_root_super_admin_can_still_be_demoted_and_self_deleted(client, auth_headers, attacker):
    second, headers = attacker

    # While it is still a super admin, self-deletion keeps the OLD 422 (the
    # self-guard), not the root's 403 — the root lock is email-scoped.
    refused = client.delete(f"{API}/users/me", headers=headers)
    assert refused.status_code == 422, refused.text
    assert refused.json()["error"]["code"] == "self_modification_forbidden"

    demote = client.patch(
        f"{API}/users/{second['id']}", json={"role": "admin"}, headers=auth_headers
    )
    assert demote.status_code == 200, demote.text
    assert demote.json()["role"] == "admin"

    # Demoted, it is an ordinary account and self-deletes normally (204):
    # nothing about the root rule touches non-root rows.
    assert client.delete(f"{API}/users/me", headers=headers).status_code == 204


def test_plain_accounts_are_still_fully_manageable(client, auth_headers):
    """Regression: the guards change nothing for everyone else."""
    created = client.post(
        f"{API}/users",
        json={"email": "plain@team.test", "role": "user", "password": "longenough"},
        headers=auth_headers,
    )
    assert created.status_code == 201, created.text
    account = created.json()

    deactivated = client.patch(
        f"{API}/users/{account['id']}", json={"is_active": False}, headers=auth_headers
    )
    assert deactivated.status_code == 200, deactivated.text
    assert deactivated.json()["is_active"] is False
    client.patch(
        f"{API}/users/{account['id']}", json={"is_active": True}, headers=auth_headers
    )

    headers = _login(client, "plain@team.test", "longenough")
    resp = client.request("DELETE", f"{API}/users/me", json={"confirm": True}, headers=headers)
    assert resp.status_code == 204, resp.text


# ------------------------------------------------- the list says which row is root


def test_the_users_list_flags_the_root_row(client, auth_headers, attacker):
    """`is_root` is computed per row so the console can render the lock.

    The flag mirrors the same `_is_root` the PATCH/DELETE guards use — only
    the seed-email row is true; every other account, super admin included.
    """
    second, _ = attacker
    resp = client.get(f"{API}/users", params={"page_size": 100}, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    flags = {row["email"]: row["is_root"] for row in resp.json()["items"]}
    assert flags[ADMIN_EMAIL] is True
    assert flags[SECOND_SUPER] is False


# ------------------------------------------------ POST cannot mint a second root


def test_post_with_the_root_email_cannot_overwrite_the_root(client, auth_headers):
    """The duplicate guard makes the seed path the only writer of that email.

    `POST /users` checks uniqueness before anything else and never updates an
    existing row, so even a request carrying an explicit `super_admin` role
    for the root's email ends in a 409 with the root's row untouched.
    """
    resp = client.post(
        f"{API}/users",
        json={"email": ADMIN_EMAIL.upper(), "role": "super_admin", "password": "hijack-pass"},
        headers=auth_headers,
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "duplicate_email"

    root = _root_row()
    assert root.role == "super_admin"
    assert root.is_active is True
    # The original password still works; the attempted one does not.
    assert _login(client, ADMIN_EMAIL, ADMIN_PASSWORD)
    hijack = client.post(
        f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": "hijack-pass"}
    )
    assert hijack.status_code == 401
