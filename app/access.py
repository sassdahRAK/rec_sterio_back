"""Folder access control shared by the console, the dataset and the guest portal.

The rules, in one place so no route drifts:

* an **admin** (the bootstrap operator) may read and manage every folder;
* the **owner** of a folder may manage it; an invited member holds exactly the
  role they were given — viewer < commenter < editor;
* a **guest** holds the role their passcode or share link was redeemed for,
  and only on that folder;
* a folder marked **public** is readable by anyone, including a visitor with no
  token at all (instruction.txt #3);
* access to a folder applies to its **subtree**, so a superset folder can be
  browsed and exported as one dataset (#6);
* a **video** inherits the role of its folder; the account that ingested it
  keeps access even before it is filed.
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import Principal
from app.errors import AppError
from app.models import AdminUser, Folder, FolderMember, Video
from app.roles import (
    ADMIN_TIER,
    OPERATOR_ROLES,
    ROLE_SUPER_ADMIN,
    TRANSCRIPT_EDIT_ROLES,
)

# Role strength: "editor can do anything a viewer can" needs no special cases.
ROLE_RANK = {"viewer": 1, "commenter": 2, "editor": 3}


def folder_or_404(db: Session, folder_id: UUID) -> Folder:
    folder = db.get(Folder, folder_id)
    if folder is None:
        raise AppError(404, "not_found", "Folder not found.", [{"id": str(folder_id)}])
    return folder


def membership(db: Session, folder_id: UUID, user_id: UUID) -> Optional[FolderMember]:
    return db.execute(
        select(FolderMember).where(
            FolderMember.folder_id == folder_id, FolderMember.user_id == user_id
        )
    ).scalar_one_or_none()


def user_for(db: Session, principal: Principal) -> Optional[AdminUser]:
    if not principal.email:
        return None
    return db.execute(
        select(AdminUser).where(AdminUser.email == principal.email)
    ).scalar_one_or_none()


def _strongest(*roles: Optional[str]) -> Optional[str]:
    best: Optional[str] = None
    for role in roles:
        if role is None:
            continue
        if best is None or ROLE_RANK.get(role, 0) > ROLE_RANK.get(best, 0):
            best = role
    return best


def folder_chain(db: Session, folder: Folder) -> list[Folder]:
    """The folder followed by its ancestors, root last."""
    chain: list[Folder] = []
    seen: set[UUID] = set()
    current: Optional[Folder] = folder
    while current is not None and current.id not in seen:
        seen.add(current.id)
        chain.append(current)
        current = db.get(Folder, current.parent_id) if current.parent_id else None
    return chain


def role_on_folder(
    db: Session, folder: Folder, principal: Principal, user: Optional[AdminUser] = None
) -> Optional[str]:
    """The caller's effective role on `folder`, or None when it is invisible."""
    if principal.role in ADMIN_TIER:
        return "editor"
    if principal.role == "telegram":
        return "editor"

    if user is None:
        user = user_for(db, principal)
    role: Optional[str] = None

    for node in folder_chain(db, folder):
        if str(node.id) in principal.scoped_folder_ids:
            role = _strongest(role, principal.role_on(node.id) or "viewer")
        if node.visibility == "public":
            role = _strongest(role, "viewer")
        if user is not None:
            if node.owner_id == user.id:
                role = _strongest(role, "editor")
            else:
                member = membership(db, node.id, user.id)
                if member is not None:
                    role = _strongest(role, member.role)
    return role


def visible_folders(db: Session, principal: Principal) -> dict[UUID, str]:
    """Every folder the caller can see, with the role they hold on it.

    One query for the folders, one for memberships — cheap for a whole page.
    """
    folders = list(db.execute(select(Folder)).scalars())
    if principal.role in ADMIN_TIER + ("telegram",):
        return {f.id: "editor" for f in folders}

    user = user_for(db, principal)
    memberships: dict[UUID, str] = {}
    if user is not None:
        memberships = {
            m.folder_id: m.role
            for m in db.execute(
                select(FolderMember).where(FolderMember.user_id == user.id)
            ).scalars()
        }
    by_id = {f.id: f for f in folders}
    out: dict[UUID, str] = {}

    for fid, folder in by_id.items():
        role: Optional[str] = None
        seen: set[UUID] = set()
        current: Optional[Folder] = folder
        while current is not None and current.id not in seen:
            seen.add(current.id)
            if str(current.id) in principal.scoped_folder_ids:
                role = _strongest(role, principal.role_on(current.id) or "viewer")
            if current.visibility == "public":
                role = _strongest(role, "viewer")
            if user is not None:
                if current.owner_id == user.id:
                    role = _strongest(role, "editor")
                role = _strongest(role, memberships.get(current.id))
            current = by_id.get(current.parent_id) if current.parent_id else None
        if role is not None:
            out[fid] = role
    return out


def video_role(
    db: Session, video: Video, principal: Principal, user: Optional[AdminUser] = None
) -> Optional[str]:
    """The role the caller holds on a video (its folder decides, unless it is theirs)."""
    if principal.role in ADMIN_TIER:
        return "editor"
    if principal.role == "telegram":
        return "editor"
    if video.created_by == principal.actor:
        return "editor"
    if video.folder_id is None:
        return None
    folder = db.get(Folder, video.folder_id)
    if folder is None:
        return None
    return role_on_folder(db, folder, principal, user=user)


# ------------------------------------------------------------------ guards


def can_read_folder(db: Session, folder: Folder, principal: Principal) -> bool:
    return role_on_folder(db, folder, principal) is not None


def can_manage_folder(db: Session, folder: Folder, principal: Principal) -> bool:
    role = role_on_folder(db, folder, principal)
    return role is not None and ROLE_RANK.get(role, 0) >= ROLE_RANK["editor"]


def require_read(db: Session, folder: Folder, principal: Principal) -> str:
    role = role_on_folder(db, folder, principal)
    if role is None:
        # Deliberately the same shape as a missing folder: whether a private
        # folder exists is not something an unauthorised caller should learn.
        raise AppError(
            404,
            "not_found",
            "Folder not found. It may be private — sign in or unlock it with a passcode.",
            [{"id": str(folder.id)}],
        )
    return role


def require_manage(db: Session, folder: Folder, principal: Principal) -> str:
    role = role_on_folder(db, folder, principal)
    if role is None or ROLE_RANK.get(role, 0) < ROLE_RANK["editor"]:
        raise AppError(
            403,
            "forbidden",
            "You need edit access to this folder.",
            [{"folder_id": str(folder.id), "role": role}],
        )
    return role


def require_comment(db: Session, folder: Folder, principal: Principal) -> str:
    role = role_on_folder(db, folder, principal)
    if role is None or ROLE_RANK.get(role, 0) < ROLE_RANK["commenter"]:
        raise AppError(
            403,
            "forbidden",
            "You need comment access to this folder.",
            [{"folder_id": str(folder.id), "role": role}],
        )
    return role


def require_operator(principal: Principal) -> None:
    """The operator surface: the job queue, global aggregates, retry and cancel.

    Jobs and stats describe the whole system rather than one folder, so there is
    no per-folder role to inherit the way `transcripts.py` does. They belong to
    the operator — the console and the bot's admin-gated commands — and a
    member who can see one folder must not read the queue that serves them all.
    """
    if principal.role not in (OPERATOR_ROLES + ("telegram",)):
        raise AppError(
            403,
            "forbidden",
            "This is an operator-only resource.",
            [{"role": principal.role}],
        )


def require_super_admin(principal: Principal) -> None:
    """User management: creating accounts and changing roles re-writes who may
    do what, so it belongs to the one account that owns the deployment —
    not to every admin, or the role system could vote itself wider."""
    if principal.role != ROLE_SUPER_ADMIN:
        raise AppError(
            403,
            "forbidden",
            "Only the super admin manages users.",
            [{"role": principal.role}],
        )


def require_video(
    db: Session, video: Video, principal: Principal, need: str = "viewer"
) -> str:
    role = video_role(db, video, principal)
    if role is None:
        raise AppError(
            403,
            "forbidden",
            "You do not have access to this video.",
            [{"video_id": str(video.id)}],
        )
    if ROLE_RANK.get(role, 0) < ROLE_RANK.get(need, 0):
        raise AppError(
            403,
            "forbidden",
            f"This action needs the {need} role on the video's folder.",
            [{"video_id": str(video.id), "role": role, "required": need}],
        )
    return role


def require_transcript_editor(
    db: Session, video: Video, principal: Principal
) -> str:
    """The transcript-edit routes: `require_video(need="editor")`, widened for
    the transcript-edit roles (`app.roles`).

    Reviewing a dataset means correcting its transcripts, so a reviewer may
    edit the transcript of *any* video, whatever folder it lives in — the
    per-folder role says nothing useful about who should fix a wrong word.
    Everything heavier (moving or deleting the media, ingesting into a
    folder) keeps the plain editor requirement, so this widening stops at
    the transcript text.
    """
    if principal.role in TRANSCRIPT_EDIT_ROLES:
        return "editor"
    return require_video(db, video, principal, need="editor")
