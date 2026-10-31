"""Public dataset surface: what a visitor sees without an account.

Instruction.txt #3 (guest views public datasets), #4 (guest opens a private
folder with the owner's passcode), #5/#6 (download the dataset as a ZIP) and #8
(cleaned data shown as cards).

Everything here is readable by an anonymous visitor; a *private* folder becomes
readable only when the caller presents a token scoped to it, obtained from a
passcode or a share link. The access check lives in `app.access`, so this router
cannot accidentally widen it.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.access import (
    can_read_folder,
    folder_or_404,
    masks_public_folders,
    require_read,
    role_on_folder,
)
from app.auth import Principal, get_optional_principal, issue_folder_token
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import Folder, Transcript, Video
from app.schemas import (
    DataCardOut,
    GuestTokenResponse,
    PasscodeRedeem,
    PublicFolderListOut,
    PublicFolderOut,
    ShareRedeem,
)
from app.security import client_ip, enforce_rate_limit
from app.services import storage, zip_export
from app.services.storage import cloud_storage_available
from app.services.zip_export import safe_name

router = APIRouter(prefix="/public", tags=["public"])

CARD_PREVIEW_CHARS = 240


def _video_cards(db: Session, folder: Folder) -> list[DataCardOut]:
    """One card per video: media, audio and transcript, with a Raw/Cleaned badge."""
    videos = db.execute(
        select(Video)
        .where(Video.folder_id == folder.id)
        .order_by(Video.position.asc(), Video.created_at.asc())
    ).scalars().all()

    cards: list[DataCardOut] = []
    for video in videos:
        transcript = zip_export.current_transcript(db, video.id)
        text = zip_export.transcript_text(transcript)
        preview = " ".join(text.split())[:CARD_PREVIEW_CHARS] or None
        cards.append(
            DataCardOut(
                video_id=video.id,
                title=video.title,
                source_url=video.source_url,
                status=video.status,
                duration_seconds=video.duration_seconds,
                thumbnail_url=video.thumbnail_url,
                folder_id=folder.id,
                folder_name=folder.name,
                # instruction.txt #8: the badge tells a visitor whether a human
                # has corrected this transcript or it is still raw ASR output.
                badge="cleaned" if (transcript is not None and transcript.cleaned) else "raw",
                transcript_id=transcript.id if transcript is not None else None,
                transcript_preview=preview,
                audio_available=storage.audio_file(video.id).exists(),
                storage_backend=video.storage_backend,
                storage_available=cloud_storage_available(video),
                media_url=f"/api/v1/videos/{video.id}/media",
                audio_url=f"/api/v1/videos/{video.id}/audio",
                transcript_url=f"/api/v1/videos/{video.id}/transcript",
            )
        )
    return cards


def _public_tree(
    db: Session, folders: list[Folder], principal: Principal
) -> list[PublicFolderOut]:
    nodes = {
        f.id: PublicFolderOut(
            id=f.id,
            name=f.name,
            description=f.description,
            visibility=f.visibility,
            my_role=role_on_folder(db, f, principal),
        )
        for f in folders
    }
    roots: list[PublicFolderOut] = []
    for folder in folders:
        node = nodes[folder.id]
        node.cards = _video_cards(db, folder)
        if folder.parent_id is not None and folder.parent_id in nodes:
            nodes[folder.parent_id].children.append(node)
        else:
            roots.append(node)
    return roots


def _subtree(db: Session, root: Folder) -> list[Folder]:
    """`root` plus every descendant, in display order.

    A folder detail view must include its sub-folders, otherwise the ZIP export
    layout (#6) and the folder tree disagree about what the folder contains.
    """
    folders = list(
        db.execute(
            select(Folder).order_by(Folder.position.asc(), Folder.created_at.asc())
        ).scalars()
    )
    wanted = {root.id}
    grew = True
    while grew:
        grew = False
        for folder in folders:
            if folder.parent_id in wanted and folder.id not in wanted:
                wanted.add(folder.id)
                grew = True
    return [folder for folder in folders if folder.id in wanted]


def _visible_public_folders(db: Session, parent_id: Optional[UUID] = None) -> list[Folder]:
    """Public folders, as a whole forest (or one level of it).

    The portal renders the tree in one pass, so the root request must include
    every descendant — filtering to top-level rows would leave `children` empty.
    """
    stmt = select(Folder).where(Folder.visibility == "public")
    if parent_id is not None:
        stmt = stmt.where(Folder.parent_id == parent_id)
    stmt = stmt.order_by(Folder.position.asc(), Folder.created_at.asc())
    return list(db.execute(stmt).scalars().all())


@router.get("/folders", response_model=PublicFolderListOut)
def list_public_folders(
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> PublicFolderListOut:
    """Public folders, plus any private folder the caller holds a token for."""
    # A portal-link token names one video and cannot list (the same rule
    # `visible_folders` applies). This tree embeds each folder's cards, so
    # without this the public library would still be enumerable through it.
    # Anonymous and folder-scoped callers are untouched.
    if masks_public_folders(principal):
        return PublicFolderListOut(items=[], total=0)
    folders = _visible_public_folders(db)
    if principal.is_guest and principal.scoped_folder_ids:
        scoped_roots = db.execute(
            select(Folder).where(Folder.id.in_(list(principal.scoped_folder_ids)))
        ).scalars().all()
        # A token on a superset folder grants its whole subtree, so the tree the
        # visitor sees must include the descendants too (#6).
        scoped: list[Folder] = []
        for root in scoped_roots:
            scoped.extend(_subtree(db, root))
        known = {f.id for f in folders}
        folders.extend(f for f in scoped if f.id not in known)
    tree = _public_tree(db, folders, principal)
    # `total` must count the roots the client actually receives, not the flat
    # forest that was fetched to build them. Otherwise a tree with one public
    # child reports `total: 2` alongside a single-item `items` list.
    return PublicFolderListOut(items=tree, total=len(tree))


@router.get("/folders/{folder_id}", response_model=PublicFolderOut)
def get_public_folder(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> PublicFolderOut:
    """A folder's cards and sub-folders. Private folders need a scoped token."""
    folder = folder_or_404(db, folder_id)
    require_read(db, folder, principal)
    return _public_tree(db, _subtree(db, folder), principal)[0]


@router.get("/folders/{folder_id}/zip")
def download_folder_zip(
    folder_id: UUID,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> FileResponse:
    """The whole folder tree as one ZIP of video + audio + transcript (#5, #6)."""
    folder = folder_or_404(db, folder_id)
    require_read(db, folder, principal)

    fd, raw_path = tempfile.mkstemp(prefix="recsterio-export-", suffix=".zip")
    os.close(fd)
    path = Path(raw_path)
    try:
        zip_export.build_folder_zip(db, folder, path)
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise AppError(409, "conflict", str(exc)) from exc
    except Exception:
        path.unlink(missing_ok=True)
        raise

    background.add_task(path.unlink, True)
    filename = f"{safe_name(folder.name, 'dataset')}.zip"
    return FileResponse(
        path,
        media_type="application/zip",
        filename=filename,
        background=background,
    )


# ------------------------------------------------------------------ access


@router.post("/passcodes/redeem", response_model=GuestTokenResponse)
def redeem_passcode_code(
    payload: PasscodeRedeem,
    request: Request,
    db: Session = Depends(get_db),
) -> GuestTokenResponse:
    """Open a private folder with its passcode and receive a scoped token."""
    from app.routers.passcodes import redeem_passcode

    # Redemption is the public guess surface: the code space is small enough
    # that an unthrottled visitor could walk it. Keyed per IP and per target
    # (folder id, typed folder name, or "any"), so one visitor cannot exhaust
    # another's attempts by hammering a different folder.
    # The key must be the same string the lookup matches on: `redeem_passcode`
    # compares folder names with `.strip().lower()`, so "Secret", "secret " and
    # "SECRET" all hit the same rows and must share one bucket — otherwise a
    # caller mints a fresh throttle by re-spelling the target.
    if payload.folder_id:
        target = str(payload.folder_id)
    else:
        target = (payload.folder_name or "").strip().lower() or "any"
    enforce_rate_limit(
        f"redeem:passcode:{client_ip(request)}:{target}",
        settings.public_redeem_rate_limit,
        settings.public_redeem_rate_window,
    )
    folder = redeem_passcode(
        db,
        payload.code,
        payload.device_key,
        folder_id=payload.folder_id,
        folder_name=payload.folder_name,
    )
    # A passcode opens the door read-only: `access` records how it was opened,
    # while the granted *role* must stay in the viewer/commenter/editor scale.
    token, expires = issue_folder_token(folder.id, "viewer")
    return GuestTokenResponse(
        token=token,
        expires_at=expires,
        folder_id=folder.id,
        role="guest",
        folder_role="viewer",
        access="passcode",
    )


@router.post("/shares/{token}/redeem", response_model=GuestTokenResponse)
def redeem_share_token(
    token: str,
    payload: ShareRedeem,
    request: Request,
    db: Session = Depends(get_db),
) -> GuestTokenResponse:
    """Open a folder through a share link and receive a scoped token."""
    from app.routers.shares import find_active_share

    # Same throttle as passcode redemption. It keys on the CLIENT IP only:
    # the token itself is an unguessable secret, so per-token buckets would
    # let a guesser spread attempts across fresh keys — the repeatable target
    # here is the caller, not the link.
    enforce_rate_limit(
        f"redeem:share:{client_ip(request)}",
        settings.public_redeem_rate_limit,
        settings.public_redeem_rate_window,
    )
    share = find_active_share(db, token)
    if share is None:
        # Same answer as an unknown link: a dead link reveals nothing.
        raise AppError(404, "not_found", "This share link is not valid.")
    folder = folder_or_404(db, share.folder_id)
    # The share link's own role decides the scope, so an "editor" link can edit.
    folder_role = share.role or "viewer"
    scoped_token, expires = issue_folder_token(folder.id, folder_role)
    return GuestTokenResponse(
        token=scoped_token,
        expires_at=expires,
        folder_id=folder.id,
        role="guest",
        folder_role=folder_role,
        access="share",
    )


@router.get("/config")
def public_config() -> dict:
    """What the portal needs to render its sign-in options.

    The Google client id is public by design (it appears in the browser URL), so
    advertising it is safe; the secret never leaves the server.
    """
    return {
        "google_client_id": settings.google_client_id or "",
        "google_enabled": bool(settings.google_client_id),
        "guest_enabled": True,
    }
