"""User management: the super admin's account console.

Creating accounts and changing roles re-writes what other people may do, so
this surface is the super admin's alone (`access.require_super_admin`) — a
plain admin can run the dataset but must not be able to mint another admin.

Because roles are database-authoritative (`auth._resolve_principal` re-reads
the row on every request), a role changed here applies to the target's very
next request. There is no token to revoke and no sign-out to force: the JWT
claim is informational, and the database wins.
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.access import require_super_admin
from app.auth import Principal, get_principal, hash_password
from app.db import get_db
from app.errors import AppError
from app.models import AdminUser
from app.schemas import AdminUserListOut, AdminUserOut, UserCreateIn, UserPatchIn

router = APIRouter(prefix="/users", tags=["users"])


def _user_or_404(db: Session, user_id: UUID) -> AdminUser:
    user = db.get(AdminUser, user_id)
    if user is None:
        raise AppError(404, "not_found", "No user with this id.", [{"id": str(user_id)}])
    return user


@router.get("", response_model=AdminUserListOut)
def list_users(
    q: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> AdminUserListOut:
    """Every account, filterable — the roster the management console renders."""
    require_super_admin(principal)

    stmt = select(AdminUser)
    count_stmt = select(func.count()).select_from(AdminUser)
    if q:
        pattern = f"%{q.strip()}%"
        condition = or_(AdminUser.email.ilike(pattern), AdminUser.name.ilike(pattern))
        stmt = stmt.where(condition)
        count_stmt = count_stmt.where(condition)
    if role is not None:
        stmt = stmt.where(AdminUser.role == role)
        count_stmt = count_stmt.where(AdminUser.role == role)
    if is_active is not None:
        stmt = stmt.where(AdminUser.is_active.is_(is_active))
        count_stmt = count_stmt.where(AdminUser.is_active.is_(is_active))

    total = db.execute(count_stmt).scalar_one()
    users = (
        db.execute(
            stmt.order_by(AdminUser.created_at.asc(), AdminUser.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .scalars()
        .all()
    )
    return AdminUserListOut(
        items=[AdminUserOut.model_validate(u) for u in users],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=AdminUserOut, status_code=201)
def create_user(
    payload: UserCreateIn,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> AdminUserOut:
    """Create an account directly, with a local password.

    Google sign-in self-serves plain users; this route exists for everything
    else — a colleague who needs the developer or reviewer role before they
    have ever signed in. The schema has already normalized the email and
    checked the role and the password floor, so only the uniqueness race is
    left here.
    """
    require_super_admin(principal)
    existing = db.execute(
        select(AdminUser).where(AdminUser.email == payload.email)
    ).scalar_one_or_none()
    if existing is not None:
        raise AppError(
            409, "duplicate_email", "An account with this email already exists.",
            [{"field": "email", "email": payload.email}],
        )

    user = AdminUser(
        email=payload.email,
        name=payload.name.strip() if payload.name else None,
        role=payload.role,
        password_hash=hash_password(payload.password),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return AdminUserOut.model_validate(user)


@router.patch("/{user_id}", response_model=AdminUserOut)
def update_user(
    user_id: UUID,
    payload: UserPatchIn,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> AdminUserOut:
    """Change a role, activate or deactivate, rename, or reset a password.

    The self-guard compares against the principal's own row id (not its email
    string): the one account that must not be demoted or deactivated here is
    the super admin doing the editing, or the deployment locks itself out of
    its own management surface. Renaming yourself or resetting your own
    password is fine.

    A changed role takes effect on the target's NEXT request: roles are read
    from the database, not from the JWT (see `auth._resolve_principal`), so
    there is no token to invalidate.
    """
    require_super_admin(principal)
    user = _user_or_404(db, user_id)
    data = payload.model_dump(exclude_unset=True)
    if not data:
        # A PATCH with no fields is a client bug, not a no-op.
        raise AppError(
            422, "validation_failed", "Provide at least one field to update.",
            [{"field": "body", "issue": "empty"}],
        )

    # `exclude_unset` tells "absent" from "explicitly null", and a null here is
    # never meaningful: `role` is NOT NULL in the database (so it would surface
    # as a 500), and `is_active: null` would coerce to False and silently
    # deactivate an account. Reject both as client errors instead.
    for field in ("role", "is_active"):
        if field in data and data[field] is None:
            raise AppError(
                422,
                "validation_failed",
                f"{field} cannot be null.",
                [{"field": field, "issue": "null"}],
            )

    caller = db.execute(
        select(AdminUser).where(AdminUser.email == principal.email)
    ).scalar_one_or_none()
    touches_privilege = "role" in data or "is_active" in data
    if touches_privilege and caller is not None and user.id == caller.id:
        raise AppError(
            422,
            "self_modification_forbidden",
            "You cannot change your own role or active state.",
            [{"field": "role", "issue": "self_modification"}],
        )

    if "role" in data:
        user.role = data["role"]
    if "is_active" in data:
        user.is_active = bool(data["is_active"])
    if "name" in data:
        user.name = data["name"].strip() if data["name"] else None
    if "password" in data:
        user.password_hash = hash_password(data["password"])

    db.commit()
    db.refresh(user)
    return AdminUserOut.model_validate(user)