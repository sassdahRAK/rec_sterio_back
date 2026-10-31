"""User management (`/users`): the super admin's account console.

Three properties are worth pinning, because each is a way this surface could
quietly become a privilege-escalation route:

* **Only the super admin may call it.** A plain admin runs the dataset; it must
  not be able to mint another admin. A signed-in `user`, a `developer`, a
  `reviewer` and an anonymous guest are all refused.
* **Privilege changes are immediate.** Roles are database-authoritative, so a
  demotion takes effect on the target's *existing* token — there is nothing to
  revoke and no re-login to wait for. Deactivation locks the token out entirely.
* **An account cannot edit its own privileges.** The super admin doing the
  editing is the one account that must not be demoted or deactivated, or the
  deployment locks itself out of its own management page.
"""
from __future__ import annotations

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AdminUser
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD
from tests.helpers import ensure_user, guest_headers, user_headers

API = "/api/v1"


def _login(client, email: str, password: str):
    return client.post(f"{API}/auth/login", json={"email": email, "password": password})


def _account(client, headers, *, email="colleague@test.local", role="user", password="secret-pass",
             name=None) -> dict:
    body = {"email": email, "role": role, "password": password}
    if name is not None:
        body["name"] = name
    resp = client.post(f"{API}/users", json=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _me(client, headers) -> dict:
    resp = client.get(f"{API}/auth/me", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------------------------------------------ the roster


def test_the_bootstrap_operator_is_a_super_admin(client, auth_headers):
    resp = client.get(f"{API}/users", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    operator = next(u for u in body["items"] if u["email"] == ADMIN_EMAIL)
    assert operator["role"] == "super_admin"
    assert operator["is_active"] is True
    assert "password_hash" not in operator
    assert body["total"] == len(body["items"])


def test_listing_filters_by_role_search_and_active_state(client, auth_headers):
    _account(client, auth_headers, email="dev@team.test", role="developer", name="Pipeline Dev")
    _account(client, auth_headers, email="rev@team.test", role="reviewer", name="Reviewer")
    retired = _account(client, auth_headers, email="gone@team.test")
    client.patch(f"{API}/users/{retired['id']}", json={"is_active": False}, headers=auth_headers)

    by_role = client.get(f"{API}/users", params={"role": "developer"}, headers=auth_headers).json()
    assert [u["email"] for u in by_role["items"]] == ["dev@team.test"]

    by_search = client.get(f"{API}/users", params={"q": "reviewer"}, headers=auth_headers).json()
    assert [u["email"] for u in by_search["items"]] == ["rev@team.test"]

    inactive = client.get(f"{API}/users", params={"is_active": "false"}, headers=auth_headers).json()
    assert [u["email"] for u in inactive["items"]] == ["gone@team.test"]
    assert inactive["total"] == 1


def test_listing_paginates(client, auth_headers):
    for n in range(3):
        _account(client, auth_headers, email=f"page{n}@team.test")
    first = client.get(f"{API}/users", params={"page": 1, "page_size": 2}, headers=auth_headers).json()
    assert len(first["items"]) == 2
    assert first["total"] == 4  # three new accounts + the bootstrap operator


# ------------------------------------------------------------------ creating


def test_a_created_account_can_sign_in_with_its_password(client, auth_headers):
    created = _account(client, auth_headers, email="New.Person@Team.TEST", password="longenough")

    # Normalized on the way in, so case cannot create a second account.
    assert created["email"] == "new.person@team.test"
    assert created["role"] == "user"
    assert created["is_active"] is True

    resp = _login(client, "new.person@team.test", "longenough")
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "user"


def test_a_duplicate_email_is_refused_with_a_code_the_console_maps(client, auth_headers):
    _account(client, auth_headers, email="taken@team.test")
    resp = client.post(
        f"{API}/users",
        json={"email": "TAKEN@team.test", "role": "user", "password": "longenough"},
        headers=auth_headers,
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["error"]["code"] == "duplicate_email"


def test_an_unknown_role_and_a_short_password_are_refused(client, auth_headers):
    bad_role = client.post(
        f"{API}/users",
        json={"email": "x@team.test", "role": "owner", "password": "longenough"},
        headers=auth_headers,
    )
    assert bad_role.status_code == 422, bad_role.text
    assert bad_role.json()["error"]["code"] == "validation_failed"

    short = client.post(
        f"{API}/users",
        json={"email": "y@team.test", "role": "user", "password": "short"},
        headers=auth_headers,
    )
    assert short.status_code == 422, short.text


def test_an_email_without_an_at_sign_is_refused(client, auth_headers):
    resp = client.post(
        f"{API}/users",
        json={"email": "not-an-address", "role": "user", "password": "longenough"},
        headers=auth_headers,
    )
    assert resp.status_code == 422, resp.text


# ------------------------------------------------------- who may call it at all


def test_only_the_super_admin_may_manage_users(client, auth_headers):
    """A guest and a plain user are refused; the list is not a public roster."""
    # The account row must exist, or the token resolves to a 401 before any
    # role is even considered — the suite's convention is ensure_user first.
    for headers in (guest_headers(), user_headers(ensure_user())):
        assert client.get(f"{API}/users", headers=headers).status_code == 403
        assert (
            client.post(
                f"{API}/users",
                json={"email": "sneak@team.test", "role": "admin", "password": "longenough"},
                headers=headers,
            ).status_code
            == 403
        )

    assert client.get(f"{API}/users").status_code == 401


def test_a_developer_and_a_reviewer_cannot_manage_users(client, auth_headers):
    dev = _account(client, auth_headers, email="dev2@team.test", role="developer")
    rev = _account(client, auth_headers, email="rev2@team.test", role="reviewer")

    for account in (dev, rev):
        token = _login(client, account["email"], "secret-pass").json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get(f"{API}/users", headers=headers).status_code == 403
        # And they cannot promote themselves, the obvious escalation attempt.
        assert (
            client.patch(
                f"{API}/users/{account['id']}", json={"role": "admin"}, headers=headers
            ).status_code
            == 403
        )


def test_a_plain_admin_cannot_mint_another_admin(client, auth_headers):
    admin = _account(client, auth_headers, email="ops@team.test", role="admin")
    token = _login(client, admin["email"], "secret-pass").json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get(f"{API}/users", headers=headers).status_code == 403
    resp = client.post(
        f"{API}/users",
        json={"email": "another@team.test", "role": "admin", "password": "longenough"},
        headers=headers,
    )
    assert resp.status_code == 403, resp.text


# ------------------------------------------------------- privilege changes


def test_a_role_change_applies_to_an_existing_token(client, auth_headers):
    account = _account(client, auth_headers, email="promote@team.test")
    token = _login(client, account["email"], "secret-pass").json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert _me(client, headers)["role"] == "user"

    resp = client.patch(
        f"{API}/users/{account['id']}", json={"role": "reviewer"}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "reviewer"

    # Same token, no re-login: the row is the source of truth.
    assert _me(client, headers)["role"] == "reviewer"


def test_a_demotion_revokes_the_operator_surface_for_the_existing_token(client, auth_headers):
    """The queue closes between two requests on the SAME token.

    A developer may operate the pipeline; demoted to a plain user mid-session,
    the token it still holds must stop reaching GET /jobs immediately — roles
    are read from the database on every request, not from the JWT.
    """
    dev = _account(client, auth_headers, email="ops-dev@team.test", role="developer")
    token = _login(client, dev["email"], "secret-pass").json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get(f"{API}/jobs", headers=headers).status_code == 200

    demoted = client.patch(
        f"{API}/users/{dev['id']}", json={"role": "user"}, headers=auth_headers
    )
    assert demoted.status_code == 200, demoted.text

    assert client.get(f"{API}/jobs", headers=headers).status_code == 403


def test_deactivating_an_account_locks_out_its_existing_token(client, auth_headers):
    account = _account(client, auth_headers, email="off@team.test")
    token = _login(client, account["email"], "secret-pass").json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    client.patch(f"{API}/users/{account['id']}", json={"is_active": False}, headers=auth_headers)

    # `_me` asserts 200, so the lockout is read straight off the response: the
    # account row is inactive and `_resolve_principal` refuses it.
    assert client.get(f"{API}/auth/me", headers=headers).status_code == 401
    assert _login(client, "off@team.test", "secret-pass").status_code == 401

    # And the account is still there, ready to be reactivated — not deleted.
    client.patch(f"{API}/users/{account['id']}", json={"is_active": True}, headers=auth_headers)
    assert _login(client, "off@team.test", "secret-pass").status_code == 200


def test_a_password_reset_replaces_the_old_password(client, auth_headers):
    account = _account(client, auth_headers, email="reset@team.test", password="old-password")
    assert _login(client, "reset@team.test", "old-password").status_code == 200

    resp = client.patch(
        f"{API}/users/{account['id']}", json={"password": "new-password"}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert _login(client, "reset@team.test", "old-password").status_code == 401
    assert _login(client, "reset@team.test", "new-password").status_code == 200


def test_an_explicit_null_privilege_field_is_a_422_not_a_500(client, auth_headers):
    """`{"role": null}` must not reach the database.

    `role` is NOT NULL there, so assigning None would surface as a 500
    internal_error; `is_active: null` would coerce to False and silently
    deactivate the target. A PATCH that says "no value" must change nothing.
    """
    account = _account(client, auth_headers, email="nullpatch@team.test")

    for body in ({"role": None}, {"is_active": None}):
        resp = client.patch(f"{API}/users/{account['id']}", json=body, headers=auth_headers)
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["code"] == "validation_failed"

    after = client.get(f"{API}/users", params={"q": "nullpatch"}, headers=auth_headers).json()
    assert after["items"][0]["role"] == "user"
    assert after["items"][0]["is_active"] is True


def test_a_name_can_be_set_and_cleared(client, auth_headers):
    account = _account(client, auth_headers, email="named@team.test")
    assert account["name"] is None

    named = client.patch(
        f"{API}/users/{account['id']}", json={"name": "Dara Sok"}, headers=auth_headers
    ).json()
    assert named["name"] == "Dara Sok"

    cleared = client.patch(
        f"{API}/users/{account['id']}", json={"name": None}, headers=auth_headers
    ).json()
    assert cleared["name"] is None


def test_an_empty_patch_is_refused(client, auth_headers):
    account = _account(client, auth_headers, email="noop@team.test")
    resp = client.patch(f"{API}/users/{account['id']}", json={}, headers=auth_headers)
    assert resp.status_code == 422, resp.text


def test_an_unknown_user_id_is_a_404(client, auth_headers):
    resp = client.patch(
        f"{API}/users/00000000-0000-0000-0000-000000000000",
        json={"role": "user"},
        headers=auth_headers,
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "not_found"


# ------------------------------------------------------------ self-protection


def test_the_super_admin_cannot_change_its_own_role_or_active_state(client, auth_headers):
    with SessionLocal() as db:
        me = db.execute(
            select(AdminUser).where(AdminUser.email == ADMIN_EMAIL)
        ).scalar_one()
        me_id = str(me.id)

    # The seeded operator IS the root (its email is ADMIN_EMAIL), so the
    # permanent-root lock answers before the plain self-guard would:
    # 403 root_account_locked, not 422 (the 422 path is covered for a
    # non-root super admin in tests/test_root_account.py).
    demote = client.patch(f"{API}/users/{me_id}", json={"role": "admin"}, headers=auth_headers)
    assert demote.status_code == 403, demote.text
    assert demote.json()["error"]["code"] == "root_account_locked"

    deactivate = client.patch(
        f"{API}/users/{me_id}", json={"is_active": False}, headers=auth_headers
    )
    assert deactivate.status_code == 403, deactivate.text

    # The account still works and is still the owner.
    assert _login(client, ADMIN_EMAIL, ADMIN_PASSWORD).status_code == 200
    assert _me(client, auth_headers)["role"] == "super_admin"


def test_the_super_admin_may_still_rename_itself_and_reset_its_own_password(client, auth_headers):
    with SessionLocal() as db:
        me = db.execute(
            select(AdminUser).where(AdminUser.email == ADMIN_EMAIL)
        ).scalar_one()
        me_id = str(me.id)

    renamed = client.patch(
        f"{API}/users/{me_id}", json={"name": "Operator"}, headers=auth_headers
    )
    assert renamed.status_code == 200, renamed.text
    # Put the seeded row back: the fixture keeps ADMIN_EMAIL between tests but
    # does not reset its columns, so a name left here would leak into others.
    client.patch(f"{API}/users/{me_id}", json={"name": None}, headers=auth_headers)

    reset = client.patch(
        f"{API}/users/{me_id}", json={"password": ADMIN_PASSWORD}, headers=auth_headers
    )
    assert reset.status_code == 200, reset.text
    assert _login(client, ADMIN_EMAIL, ADMIN_PASSWORD).status_code == 200


def test_seeding_promotes_an_existing_row_without_touching_its_password(client, auth_headers):
    """ADMIN_EMAIL names the deployment's one super admin, whatever the row
    used to carry — a database created before the role system existed would
    otherwise strand its own operator. Promotion is idempotent and never
    rotates the password: that stays an operator decision.
    """
    from app.auth import seed_admin_user

    with SessionLocal() as db:
        me = db.execute(
            select(AdminUser).where(AdminUser.email == ADMIN_EMAIL)
        ).scalar_one()
        before_hash = me.password_hash
        me.role = "admin"  # a pre-role-system row
        db.commit()

        promoted = seed_admin_user(db)
        assert promoted.id == me.id
        assert promoted.role == "super_admin"
        assert promoted.password_hash == before_hash

        # Idempotent: a second boot changes nothing.
        again = seed_admin_user(db)
        assert again.id == me.id
        assert again.role == "super_admin"
