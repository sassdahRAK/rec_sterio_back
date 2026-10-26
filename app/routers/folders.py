"""Folder routes: the dataset grouping that sharing and export hang off.

A folder is the unit an admin publishes (instruction.txt #1), nests into
sub-folders that mirror the ZIP layout (#6), and attaches share links,
passcodes and comments to.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.access import (
    ROLE_RANK,
    folder_or_404,
    require_manage,
    require_read,
    user_for,
    visible_folders,
)
from app.auth import Principal, get_optional_principal, get_principal
from app.db import get_db
from app.errors import AppError
from app.models import (
    AdminUser,
    Folder,
    FolderComment,
    FolderMember,
    Transcript,
    Video,
)
from app.schemas import (
    DataCardOut,
    DatasetCardListOut,
    FolderCommentCreate,
    FolderCommentListOut,
    FolderCommentOut,
    FolderCreate,
    FolderListOut,
    FolderMemberListOut,
    FolderMemberOut,
    FolderOut,
    FolderUpdate,
    MemberInvite,
    MemberUpdate,
    VideoFolderUpdate,
    VideoOut,
)
from app.services import storage, zip_export

logger = logging.getLogger("recsterio.folders")

router = APIRouter(tags=["folders"])


def _current_user(db: Session, principal: Principal) -> AdminUser:
    if not principal.is_signed_in:
        raise AppError(
            401, "unauthorized", "Sign in to manage dataset folders.",
            [{"header": "Authorization"}],
        )
    user = user_for(db, principal)
    if user is None:
        raise AppError(401, "unauthorized", "Invalid or expired token.")
    return user


def _video_count(db: Session, folder_ids: list[UUID]) -> dict[UUID, int]:
    if not folder_ids:
        return {}
    rows = db.execute(
        select(Video.folder_id, func.count())
        .where(Video.folder_id.in_(folder_ids))
        .group_by(Video.folder_id)
    ).all()
    return {row[0]: row[1] for row in rows}


def _serialize(
    db: Session,
    folder: Folder,
    counts: dict[UUID, int],
    role: Optional[str] = None,
) -> FolderOut:
    out = FolderOut.model_validate(folder)
    # `from_attributes` also reads the ORM `children` relationship, which would
    # pre-populate this list and make `_tree` append each child a second time.
    out.children = []
    out.video_count = counts.get(folder.id, 0)
    out.my_role = role
    owner = db.get(AdminUser, folder.owner_id)
    out.owner_email = owner.email if owner else None
    return out


def _tree(
    db: Session,
    folders: list[Folder],
    counts: dict[UUID, int],
    roles: dict[UUID, str],
) -> list[FolderOut]:
    """Nest the flat folder list into the shape the UI renders."""
    nodes = {f.id: _serialize(db, f, counts, roles.get(f.id)) for f in folders}
    roots: list[FolderOut] = []
    for folder in folders:
        node = nodes[folder.id]
        parent_id = folder.parent_id
        if parent_id is not None and parent_id in nodes:
            nodes[parent_id].children.append(node)
        else:
            roots.append(node)
    return roots


# ---------------------------------------------------------------- folders


@router.get("/folders", response_model=FolderListOut)
def list_folders(
    parent_id: Optional[UUID] = Query(None),
    visibility: Optional[str] = None,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> FolderListOut:
    """Every folder the caller may see. Guests get the public ones (#3)."""
    if parent_id is not None:
        require_read(db, folder_or_404(db, parent_id), principal)

    stmt = select(Folder)
    if visibility:
        stmt = stmt.where(Folder.visibility == visibility)
    if parent_id is not None:
        stmt = stmt.where(Folder.parent_id == parent_id)
    stmt = stmt.order_by(Folder.position.asc(), Folder.created_at.asc())
    folders = list(db.execute(stmt).scalars().all())

    roles = visible_folders(db, principal)
    folders = [f for f in folders if f.id in roles]
    counts = _video_count(db, [f.id for f in folders])
    items = _tree(db, folders, counts, roles)
    return FolderListOut(items=items, total=len(items))


@router.post("/folders", response_model=FolderOut, status_code=201)
def create_folder(
    payload: FolderCreate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> FolderOut:
    user = _current_user(db, principal)
    if payload.visibility == "public" and principal.role != "admin":
        # Only the operator publishes to anonymous visitors.
        raise AppError(
            403, "forbidden", "Only an admin can create a public folder.",
            [{"field": "visibility"}],
        )
    if payload.parent_id is not None:
        require_manage(db, folder_or_404(db, payload.parent_id), principal)

    sibling = (
        Folder.parent_id == payload.parent_id
        if payload.parent_id is not None
        else Folder.parent_id.is_(None)
    )
    position = db.execute(
        select(func.coalesce(func.max(Folder.position), -1) + 1).where(sibling)
    ).scalar_one()

    folder = Folder(
        name=payload.name.strip(),
        description=payload.description,
        visibility=payload.visibility,
        parent_id=payload.parent_id,
        position=payload.position or position,
        owner_id=user.id,
    )
    db.add(folder)
    db.commit()
    db.refresh(folder)
    return _serialize(db, folder, _video_count(db, [folder.id]), "editor")


@router.get("/folders/{folder_id}", response_model=FolderOut)
def get_folder(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> FolderOut:
    folder = folder_or_404(db, folder_id)
    role = require_read(db, folder, principal)
    counts = _video_count(db, [folder.id])
    out = _serialize(db, folder, counts, role)
    roles = visible_folders(db, principal)
    children = [
        c
        for c in db.execute(
            select(Folder)
            .where(Folder.parent_id == folder.id)
            .order_by(Folder.position.asc(), Folder.created_at.asc())
        ).scalars()
        if c.id in roles
    ]
    child_counts = _video_count(db, [c.id for c in children])
    out.children = [_serialize(db, c, child_counts, roles.get(c.id)) for c in children]
    return out


@router.patch("/folders/{folder_id}", response_model=FolderOut)
def update_folder(
    folder_id: UUID,
    payload: FolderUpdate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> FolderOut:
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)

    data = payload.model_dump(exclude_unset=True)
    new_parent = data.get("parent_id")
    if new_parent is not None:
        if new_parent == folder.id:
            raise AppError(
                422, "validation_failed", "A folder cannot be its own parent.",
                [{"field": "parent_id", "issue": "self_reference"}],
            )
        require_manage(db, folder_or_404(db, new_parent), principal)
    if data.get("visibility") == "public" and principal.role != "admin":
        raise AppError(
            403, "forbidden", "Only an admin can make a folder public.",
            [{"field": "visibility"}],
        )

    for field, value in data.items():
        setattr(folder, field, value)
    db.commit()
    db.refresh(folder)
    return _serialize(db, folder, _video_count(db, [folder.id]))


@router.delete("/folders/{folder_id}", status_code=204)
def delete_folder(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> None:
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)
    # Children cascade in the database; their videos are detached, not deleted,
    # so removing a grouping never destroys the underlying media.
    db.delete(folder)
    db.commit()


# ---------------------------------------------------------------- videos


@router.get("/folders/{folder_id}/videos", response_model=list[VideoOut])
def list_folder_videos(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> list[VideoOut]:
    folder = folder_or_404(db, folder_id)
    require_read(db, folder, principal)
    items = db.execute(
        select(Video)
        .where(Video.folder_id == folder_id)
        .order_by(Video.position.asc(), Video.created_at.asc())
    ).scalars().all()
    return [VideoOut.model_validate(v) for v in items]


@router.put("/videos/{video_id}/folder", response_model=VideoOut)
def place_video(
    video_id: UUID,
    payload: VideoFolderUpdate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    """Move a video into a folder (or out of one, with folder_id=null)."""
    video = db.get(Video, video_id)
    if video is None:
        raise AppError(404, "not_found", "Video not found.", [{"id": str(video_id)}])

    if payload.folder_id is None:
        video.folder_id = None
    else:
        folder = folder_or_404(db, payload.folder_id)
        require_manage(db, folder, principal)
        video.folder_id = folder.id
    if payload.position is not None:
        video.position = payload.position

    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)


# ----------------------------------------------------------- dataset cards


def _card(db: Session, video: Video, folder: Optional[Folder]) -> DataCardOut:
    transcript = zip_export.current_transcript(db, video.id)
    preview = None
    if transcript is not None and transcript.content:
        preview = " ".join(transcript.content.split())[:280]
    base = f"/api/v1/videos/{video.id}"
    return DataCardOut(
        video_id=video.id,
        title=video.title,
        source_url=video.source_url,
        status=video.status,
        duration_seconds=video.duration_seconds,
        thumbnail_url=video.thumbnail_url,
        folder_id=video.folder_id,
        folder_name=folder.name if folder else None,
        badge="cleaned" if transcript is not None and transcript.cleaned else "raw",
        transcript_id=transcript.id if transcript else None,
        transcript_preview=preview,
        audio_available=storage.audio_file(video.id).exists(),
        media_url=f"{base}/media",
        audio_url=f"{base}/audio",
        transcript_url=f"{base}/transcript",
    )


@router.get("/dataset/cards", response_model=DatasetCardListOut)
def list_dataset_cards(
    folder_id: Optional[UUID] = None,
    cleaned: Optional[bool] = None,
    q: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> DatasetCardListOut:
    """The dataset view: every video the caller may read, as a card (#8, #9).

    Guests see the public folders and nothing else; a passcode token narrows
    this to the folders it was scoped to.
    """
    roles = visible_folders(db, principal)
    if folder_id is not None:
        folder = folder_or_404(db, folder_id)
        require_read(db, folder, principal)

    if principal.role in ("admin", "telegram"):
        stmt = select(Video)
    else:
        accessible = list(roles.keys())
        conditions = [Video.created_by == principal.actor]
        if accessible:
            conditions.append(Video.folder_id.in_(accessible))
        stmt = select(Video).where(or_(*conditions))

    if folder_id is not None:
        stmt = stmt.where(Video.folder_id == folder_id)
    if q:
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(or_(Video.title.ilike(pattern), Video.source_url.ilike(pattern)))
    stmt = stmt.order_by(Video.created_at.desc())
    videos = list(db.execute(stmt).scalars())

    if cleaned is not None:
        filtered = []
        for video in videos:
            transcript = zip_export.current_transcript(db, video.id)
            if transcript is not None and bool(transcript.cleaned) is cleaned:
                filtered.append(video)
        videos = filtered

    names = {f.id: f for f in db.execute(select(Folder)).scalars()}
    total = len(videos)
    start = (page - 1) * page_size
    items = [
        _card(db, video, names.get(video.folder_id))
        for video in videos[start : start + page_size]
    ]
    return DatasetCardListOut(items=items, total=total, page=page, page_size=page_size)


# ---------------------------------------------------------------- members


def _member_out(db: Session, member: FolderMember) -> FolderMemberOut:
    user = db.get(AdminUser, member.user_id)
    return FolderMemberOut(
        user_id=member.user_id,
        email=user.email if user else "(deleted account)",
        name=user.name if user else None,
        role=member.role,
        created_at=member.created_at,
    )


@router.get("/folders/{folder_id}/members", response_model=FolderMemberListOut)
def list_members(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> FolderMemberListOut:
    folder = folder_or_404(db, folder_id)
    require_read(db, folder, principal)
    members = db.execute(
        select(FolderMember)
        .where(FolderMember.folder_id == folder_id)
        .order_by(FolderMember.created_at.asc())
    ).scalars().all()
    return FolderMemberListOut(items=[_member_out(db, m) for m in members])


@router.post("/folders/{folder_id}/members", response_model=FolderMemberOut, status_code=201)
def invite_member(
    folder_id: UUID,
    payload: MemberInvite,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> FolderMemberOut:
    """Invite by email. The account may not exist yet, so a placeholder is
    created and the invitation is waiting when they first sign in (#2)."""
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)
    email = payload.email.strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise AppError(
            422, "validation_failed", "A valid email address is required.",
            [{"field": "email"}],
        )

    user = db.execute(
        select(AdminUser).where(AdminUser.email == email)
    ).scalar_one_or_none()
    if user is None:
        user = AdminUser(email=email, role="member", password_hash=None)
        db.add(user)
        db.flush()

    member = db.execute(
        select(FolderMember).where(
            FolderMember.folder_id == folder.id, FolderMember.user_id == user.id
        )
    ).scalar_one_or_none()
    if member is None:
        member = FolderMember(folder_id=folder.id, user_id=user.id, role=payload.role)
        db.add(member)
    else:
        member.role = payload.role
    db.commit()
    db.refresh(member)
    return _member_out(db, member)


@router.patch("/folders/{folder_id}/members/{user_id}", response_model=FolderMemberOut)
def update_member(
    folder_id: UUID,
    user_id: UUID,
    payload: MemberUpdate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> FolderMemberOut:
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)
    member = db.execute(
        select(FolderMember).where(
            FolderMember.folder_id == folder_id, FolderMember.user_id == user_id
        )
    ).scalar_one_or_none()
    if member is None:
        raise AppError(404, "not_found", "This user is not a member of the folder.")
    member.role = payload.role
    db.commit()
    db.refresh(member)
    return _member_out(db, member)


@router.delete("/folders/{folder_id}/members/{user_id}", status_code=204)
def remove_member(
    folder_id: UUID,
    user_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> None:
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)
    member = db.execute(
        select(FolderMember).where(
            FolderMember.folder_id == folder_id, FolderMember.user_id == user_id
        )
    ).scalar_one_or_none()
    if member is None:
        raise AppError(404, "not_found", "This user is not a member of the folder.")
    db.delete(member)
    db.commit()


# ---------------------------------------------------------------- comments


@router.get("/folders/{folder_id}/comments", response_model=FolderCommentListOut)
def list_comments(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> FolderCommentListOut:
    folder = folder_or_404(db, folder_id)
    require_read(db, folder, principal)
    items = db.execute(
        select(FolderComment)
        .where(FolderComment.folder_id == folder_id)
        .order_by(FolderComment.created_at.asc())
    ).scalars().all()
    return FolderCommentListOut(
        items=[FolderCommentOut.model_validate(c) for c in items]
    )


@router.post("/folders/{folder_id}/comments", response_model=FolderCommentOut, status_code=201)
def add_comment(
    folder_id: UUID,
    payload: FolderCommentCreate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> FolderCommentOut:
    """Commenter role is meaningful here (instruction.txt #1)."""
    folder = folder_or_404(db, folder_id)
    role = require_read(db, folder, principal)
    if ROLE_RANK.get(role, 0) < ROLE_RANK["commenter"]:
        raise AppError(
            403, "forbidden", "You need comment access to this folder.",
            [{"folder_id": str(folder.id), "role": role}],
        )
    # A member is attributed to their account; a guest who arrived through a
    # share link is attributed to their actor string, since there is no account
    # row to point at. Without this, the "can comment" share role would be a
    # lie for exactly the people it is meant for.
    user = user_for(db, principal) if principal.is_signed_in else None
    if user is None and not principal.is_guest:
        raise AppError(401, "unauthorized", "Sign in to comment on a folder.")
    comment = FolderComment(
        folder_id=folder.id,
        author_id=user.id if user is not None else None,
        author_label=principal.actor,
        body=payload.body,
    )
    db.add(comment)
    db.commit()
    db.refresh(comment)
    return FolderCommentOut.model_validate(comment)


# ------------------------------------------------------------------ export


@router.get("/folders/{folder_id}/transcripts")
def list_folder_transcripts(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> dict:
    """Transcript metadata for a folder, used by the export preview."""
    folder = folder_or_404(db, folder_id)
    require_read(db, folder, principal)
    videos = db.execute(
        select(Video).where(Video.folder_id == folder.id)
    ).scalars().all()
    rows = []
    for video in videos:
        transcripts = db.execute(
            select(Transcript)
            .where(Transcript.video_id == video.id)
            .order_by(Transcript.version.desc())
        ).scalars().all()
        rows.append(
            {
                "video_id": str(video.id),
                "title": video.title,
                "status": video.status,
                "transcripts": [
                    {
                        "id": str(t.id),
                        "version": t.version,
                        "source": t.source,
                        "cleaned": t.cleaned,
                    }
                    for t in transcripts
                ],
            }
        )
    return {"folder_id": str(folder.id), "videos": rows}


@router.get("/folders/{folder_id}/download")
def download_folder_zip(
    folder_id: UUID,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> FileResponse:
    """One ZIP for the folder and its subtree: video + audio + transcript (#5, #6)."""
    folder = folder_or_404(db, folder_id)
    require_read(db, folder, principal)
    tmp = tempfile.NamedTemporaryFile(
        prefix="recsterio-export-", suffix=".zip", delete=False
    )
    tmp.close()
    dest = Path(tmp.name)
    try:
        zip_export.build_folder_zip(db, folder, dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise AppError(
            409,
            "nothing_to_export",
            "This folder has no downloadable media yet.",
            [{"folder_id": str(folder.id), "reason": str(exc)}],
        ) from exc

    def _cleanup(path: Path = dest) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort
            logger.warning("could not remove temp archive %s", path)

    background.add_task(_cleanup)
    return FileResponse(
        dest,
        media_type="application/zip",
        filename=f"{zip_export.safe_name(folder.name, 'dataset')}.zip",
        headers={"Cache-Control": "no-store"},
    )
