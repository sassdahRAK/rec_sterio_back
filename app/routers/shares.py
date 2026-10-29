"""Share links: a revocable URL that grants a role on a folder (instruction.txt #1).

A link is the low-friction option — no account, no code — so it grants a
read-only role by default and can be revoked without touching the folder.

The bare token exists exactly once: in the 201 response (and the URL it
builds), per the API contract. At rest the row stores `sha256(token)`, so a
database leak does not hand out live links; a presented token is hashed and
matched against that digest. Rows written before the switch hold the
plaintext — they are recognized lazily, kept working, and rehash-stored the
first time they are actually used.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.access import folder_or_404, require_manage
from app.auth import Principal, get_principal
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import Folder, ShareLink
from app.schemas import ShareLinkCreated, ShareLinkCreate, ShareLinkOut

router = APIRouter(tags=["shares"])

TOKEN_BYTES = 24


def new_share_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_digest(token: str) -> str:
    """How a share token is stored: sha256 of the presented value."""
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def _get_share(db: Session, share_id: UUID) -> ShareLink:
    share = db.get(ShareLink, share_id)
    if share is None:
        raise AppError(404, "not_found", "Share link not found.", [{"id": str(share_id)}])
    return share


def find_active_share(db: Session, token: str) -> ShareLink | None:
    """The share for `token`, or None when unknown, revoked or expired.

    Lookup is by digest. A legacy row still holding the plaintext token is
    matched by its old value and rehash-stored when it is actually active —
    the conversion happens on successful use, so no data migration is needed
    and a dead old link is left exactly as it was.
    """
    wanted = token_digest(token)
    share = db.execute(
        select(ShareLink).where(ShareLink.token == wanted)
    ).scalar_one_or_none()
    if share is None:
        # Legacy plaintext row (pre-hash at-rest storage): match directly.
        legacy = db.execute(
            select(ShareLink).where(ShareLink.token == token.strip())
        ).scalar_one_or_none()
        if legacy is None or legacy.revoked_at is not None:
            return None
        if legacy.expires_at is not None and legacy.expires_at <= datetime.now(timezone.utc):
            return None
        # ...and convert it, now that this use is successful.
        legacy.token = wanted
        db.commit()
        return legacy
    if share.revoked_at is not None:
        return None
    if share.expires_at is not None and share.expires_at <= datetime.now(timezone.utc):
        return None
    return share


def share_url(token: str) -> str:
    base = settings.portal_frontend_url.rstrip("/")
    return f"{base}/share/{token}"


@router.post("/folders/{folder_id}/shares", response_model=ShareLinkCreated, status_code=201)
def create_share(
    folder_id: UUID,
    payload: ShareLinkCreate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> ShareLinkCreated:
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)

    plain = new_share_token()
    share = ShareLink(
        folder_id=folder.id,
        # Only the digest is stored; the plaintext lives in this response and
        # the URL the caller sends out, never in the database.
        token=token_digest(plain),
        role=payload.role,
        expires_at=payload.expires_at,
        created_by=principal.actor,
    )
    db.add(share)
    db.commit()
    db.refresh(share)
    # `url` is derived, not stored, so it cannot come from `model_validate(share)`.
    out = ShareLinkOut.model_validate(share)
    # The one response that reveals the bare token (contract §Share links).
    out.token = plain
    return ShareLinkCreated(**out.model_dump(), url=share_url(plain))


@router.get("/folders/{folder_id}/shares", response_model=list[ShareLinkOut])
def list_shares(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[ShareLinkOut]:
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)
    items = db.execute(
        select(ShareLink)
        .where(ShareLink.folder_id == folder_id)
        .order_by(ShareLink.created_at.desc())
    ).scalars().all()
    return [ShareLinkOut.model_validate(s) for s in items]


@router.delete("/shares/{share_id}", status_code=204)
def revoke_share(
    share_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> None:
    share = _get_share(db, share_id)
    require_manage(db, folder_or_404(db, share.folder_id), principal)
    share.revoked_at = datetime.now(timezone.utc)
    db.commit()


@router.get("/shares/{share_id}", response_model=ShareLinkOut)
def describe_share(
    share_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> ShareLinkOut:
    """Preview a link's target so the UI can label it before sending it out."""
    share = _get_share(db, share_id)
    folder = folder_or_404(db, share.folder_id)
    require_manage(db, folder, principal)
    return ShareLinkOut.model_validate(share)


def folder_for_share(db: Session, token: str) -> Folder:
    """Resolve a share token to its folder, or raise a uniform 404.

    Unknown, revoked and expired links all look the same: a link that stops
    working should not reveal whether it ever existed.
    """
    share = find_active_share(db, token)
    if share is None:
        raise AppError(404, "not_found", "This share link is not valid.")
    return folder_or_404(db, share.folder_id)
