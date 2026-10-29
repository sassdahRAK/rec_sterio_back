"""User management: the super admin's account console.

Creating accounts and changing roles re-writes what other people may do, so
this surface is the super admin's alone (`access.require_super_admin`) — a
plain admin can run the dataset but must not be able to mint another admin.

Because roles are database-authoritative (`auth._resolve_principal` re-reads
the row on every request), a role changed here applies to the target's very
next request. There is no token to revoke and no sign-out to force: the JWT
claim is informational, and the database wins.

`DELETE /users/me` is the account-holder's own self-service deletion: it
anonymizes the row, which is ALSO the revocation mechanism for the session
token — every bearer token carries the (old) email as its `sub` claim, and
principal resolution re-reads the row by that email on every request, so a
token whose subject is now `deleted:{id}` simply fails to resolve.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session

from app.access import require_super_admin
from app.auth import Principal, get_principal, hash_password
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import (
    AdminUser,
    DriveLink,
    Folder,
    FolderComment,
    FolderMember,
    FolderPasscode,
    LinkCode,
    PasscodeGrant,
)
from app.roles import ROLE_SUPER_ADMIN
from app.schemas import (
    AdminUserListOut,
    AdminUserOut,
    UserCreateIn,
    UserDeleteIn,
    UserPatchIn,
)

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


@router.delete("/me", status_code=204)
def delete_me(
    payload: Optional[UserDeleteIn] = None,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> None:
    """The account holder deletes their own account (GDPR-style self-service).

    What happens:

    * the row's PII is anonymized in place — name and email become
      `deleted:{id}`, password, avatar, Google subject and Telegram linkage
      are cleared — and the row KEEPS its id, so attribution strings
      (`created_by` actor strings on videos/transcripts, folder comments)
      stay intact history rather than dangling references;
    * this anonymization is the token revocation: `_resolve_principal`
      resolves every account bearer by its `sub` (the email at issue time),
      so any token the account ever held stops resolving from this moment;
    * because the row survives, no FK cascade fires — the same transaction
      therefore also deletes what a hard row-deletion would have taken: the
      connected cloud links (live OAuth refresh tokens, provider account
      emails), outstanding Telegram link codes, folder memberships, and the
      device grants of passcodes this account minted (those passcodes are
      revoked, so none of them keeps opening doors);
    * dataset videos are kept for research integrity (a folder's videos
      stay with the folder, unfiled ones stay with their `created_by`
      attribution);
    * a super admin (incl. the seeded operator) can never self-delete — it is
      the deployment's management surface, mirroring the PATCH self-guard;
    * nothing happens to the bot: the Telegram service identity is not an
      account row and cannot reach this route.
    """
    payload = payload or UserDeleteIn()
    caller = db.execute(
        select(AdminUser).where(AdminUser.email == (principal.email or ""))
    ).scalar_one_or_none()
    if caller is None:
        # A guest, a bot service token or a Telegram person names no row; and
        # for the bot specifically there is nothing to delete — it is not an
        # account and must not become an anonymizable one.
        raise AppError(
            403,
            "forbidden",
            "Sign in to a console account to delete it.",
        )
    if caller.role == ROLE_SUPER_ADMIN:
        raise AppError(
            422,
            "self_modification_forbidden",
            "The super admin account cannot delete itself.",
            [{"role": caller.role}],
        )

    owned = db.execute(
        select(Folder).where(Folder.owner_id == caller.id)
    ).scalars().all()
    marker = f"deleted:{caller.id}"
    if owned and not payload.confirm:
        raise AppError(
            409,
            "confirmation_required",
            "This account still owns folders. Send {\"confirm\": true} to have "
            "them transferred to the operator and the account deleted.",
            [{"folders": len(owned)}],
        )
    if owned:
        operator = db.execute(
            select(AdminUser).where(
                AdminUser.email == (settings.admin_email or ""),
                AdminUser.role == ROLE_SUPER_ADMIN,
            )
        ).scalar_one_or_none()
        if operator is None:
            raise AppError(
                409,
                "conflict",
                "There is no operator account to transfer this account's "
                "folders to, so the deletion is refused.",
            )
        for folder in owned:
            folder.owner_id = operator.id
        # Their comments stop attributing a now-removed person; the text stays.
        db.execute(
            update(FolderComment)
            .where(FolderComment.author_id == caller.id)
            .values(author_id=None, author_label=marker)
        )
    # Nothing cascades here: the row is anonymized, never deleted, so NO
    # foreign key fires and every credential it points at would survive. The
    # hard-delete cleanups are performed inside this same transaction, and the
    # anonymization itself is what revokes the session tokens.
    created_actor = f"{caller.role}:{caller.email}"  # actor string their minted passcodes carry
    caller.email = marker          # unique — the row id guarantees no collision
    caller.name = marker
    caller.password_hash = None
    caller.avatar_url = None
    caller.google_sub = None       # unique FK to the Google identity — also unlinks
    caller.telegram_user_id = None  # unique link to a Telegram account — unlinks
    caller.role = "user"
    # Drive links hold a live OAuth refresh token plus the provider account
    # email — secrets and PII that must not outlive the account.
    db.execute(delete(DriveLink).where(DriveLink.user_id == caller.id))
    # Outstanding Telegram link codes: a still-redeemable code would let the
    # bot re-link the anonymized row.
    db.execute(delete(LinkCode).where(LinkCode.admin_user_id == caller.id))
    # Folder memberships granted to this identity: gone with it.
    db.execute(delete(FolderMember).where(FolderMember.user_id == caller.id))
    # Passcodes this account minted are doors created by a deleted identity:
    # revoke them, and take their device grants along (a grant row knows only
    # the passcode, never the user, so the passcode is the sole path here).
    theirs = select(FolderPasscode.id).where(FolderPasscode.created_by == created_actor)
    db.execute(delete(PasscodeGrant).where(PasscodeGrant.passcode_id.in_(theirs)))
    db.execute(
        update(FolderPasscode)
        .where(FolderPasscode.created_by == created_actor, FolderPasscode.revoked_at.is_(None))
        .values(revoked_at=datetime.now(timezone.utc))
    )
    db.commit()
    return None