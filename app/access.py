"""Folder access control shared by the console, the dataset and the guest portal.

The rules, in one place so no route drifts:

Capability matrix (``docs/API_CONTRACT.md`` §0)::

    | capability                        | super_admin | admin | developer | reviewer | user            |
    |-----------------------------------|-------------|-------|-----------|----------|-----------------|
    | jobs: view + retry, stats         | yes         | yes   | yes       | -        | -               |
    | library: view + act on visible    | yes         | yes   | yes       | yes      | scoped          |
    | videos: retry/delete/move/fork    | yes         | yes   | yes       | -        | -               |
    | transcripts: edit/clean any       | yes         | yes   | -         | yes      | editor-by-folder|
    | folders: create/publish/share/zip | yes         | yes   | -         | -        | by folder role  |

**A library tier is a capability, never a folder rank.** Do not seed
``developer``/``reviewer`` into :func:`role_on_folder` or
:func:`visible_folders` (the regression in commit 03639b5): a single rank
cannot express the matrix — ``editor`` would hand the developer folder
administration *and* transcript editing, both of which the matrix denies — and
it would make an ungranted private folder visible. The tier's widening lives in
the capability predicates below (:func:`can_act_on_video`,
:func:`can_edit_transcript`, :func:`can_administer_folder`); the media and
folder predicates apply only to a folder the account can already see, while
:func:`can_edit_transcript` grants its tier whatever folder the video lives in.
An ungranted private folder resolves to ``None`` for every non-admin account:
404 on read, absent from listings.

* an **admin** (the bootstrap operator) may read and manage every folder;
* the **library tier** (`developer`, `reviewer`) is **not** granted folder
  visibility: a private folder with no grant on it stays invisible to them
  (404, absent from their listings), exactly as for any other account. Their
  widening is per *capability* — `require_media_editor`/`require_media_folder`
  let a developer retry/delete/move/fork the media of a unit they can already
  see, while `require_transcript_editor` lets a reviewer correct the transcript
  of *any* video, whatever folder it lives in (the transcript grant is not
  conditioned on folder visibility; see :func:`can_edit_transcript`). Neither
  grants folder administration: create, rename,
  delete, share, passcode and member management still need `editor` on that
  folder (`require_manage`), which the library tier does not hold by role;
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
    LIBRARY_MEDIA_ROLES,
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


def masks_public_folders(principal: Principal) -> bool:
    """Whether public visibility must be withheld from this caller.

    A portal-link token names exactly one unit ("one video, cannot list"), so
    public folders must not contribute visibility: otherwise the same
    credential would read every public folder and enumerate the whole public
    library through `GET /dataset/cards`, `GET /videos` and search. The grant
    still reaches its own video — `video_role` short-circuits on
    `scoped_video_ids` before any folder rule runs.

    A folder-scoped (passcode/share) token carries an empty
    `scoped_video_ids`, so it keeps public visibility unchanged; an account
    token never carries the claim, and anonymous guests are empty too.
    """
    return principal.is_guest and bool(principal.scoped_video_ids)


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
    """The caller's effective role on `folder`, or None when it is invisible.

    Resolution is the account role plus the per-folder walk, nothing else: the
    admin tier and the bot's service identity are editors everywhere; everyone else
    starts from None and can only gain a role through a scoped token, public
    visibility, ownership or an explicit membership. The library tier
    (`developer`, `reviewer`) holds no implicit role on a folder — a private
    folder with no grant is invisible to them, exactly as to any other account.
    """
    if principal.role in ADMIN_TIER:
        return "editor"
    # Only the bot's own service identity is an editor everywhere. A person
    # acting through the bot (`telegram:<uid>`) is resolved to their linked
    # account role, or kept as an ordinary `telegram` principal with no blanket
    # grant — otherwise knowing a folder id would read someone else's dataset.
    if principal.is_bot_service:
        return "editor"

    if user is None:
        user = user_for(db, principal)
    role: Optional[str] = None
    # A portal-link token must not inherit "viewer" from a public folder; its
    # one video reaches this role through `video_role`'s scoped-id branch.
    public_counts = not masks_public_folders(principal)

    for node in folder_chain(db, folder):
        if str(node.id) in principal.scoped_folder_ids:
            role = _strongest(role, principal.role_on(node.id) or "viewer")
        if public_counts and node.visibility == "public":
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
    Everyone except the admin tier and the bot's service identity gets only
    what the per-folder rules grant: public visibility, ownership, an explicit
    membership or a scoped token. The library tier is no different here — a
    private folder they hold no role on is absent, so the folder tree never
    leaks its existence or name.
    """
    folders = list(db.execute(select(Folder)).scalars())
    if principal.role in ADMIN_TIER or principal.is_bot_service:
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
    # The same suppression `role_on_folder` applies: a portal-link token names
    # one video and must not see public folders in any listing (this map feeds
    # `/dataset/cards`, `/videos` and the folder endpoints).
    public_counts = not masks_public_folders(principal)

    for fid, folder in by_id.items():
        role: Optional[str] = None
        seen: set[UUID] = set()
        current: Optional[Folder] = folder
        while current is not None and current.id not in seen:
            seen.add(current.id)
            if str(current.id) in principal.scoped_folder_ids:
                role = _strongest(role, principal.role_on(current.id) or "viewer")
            if public_counts and current.visibility == "public":
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
    if principal.is_bot_service:
        return "editor"
    if video.created_by == principal.actor:
        return "editor"
    # The portal-link grant is checked before the unfiled early return: that
    # return is what made an unfiled bot-ingested unit invisible to the portal,
    # and a scoped token is the one credential that may name such a unit. The
    # role comes from the token's `video_roles` claim (viewer when absent) so
    # the claim the issuer writes is the claim read back here. An id outside
    # `scoped_video_ids` is still None — this never widens.
    portal_role = principal.role_on_video(video.id)
    if portal_role is not None:
        return portal_role
    if video.folder_id is None:
        return None
    folder = db.get(Folder, video.folder_id)
    if folder is None:
        return None
    return role_on_folder(db, folder, principal, user=user)


# ---------------------------------------------------- capability predicates
#
# One predicate per matrix row the library tier widens. They are the single
# source the guards below delegate to; do not reintroduce a per-folder "floor"
# for developer/reviewer (commit 03639b5) — one rank cannot express "delete the
# video, but not administer the folder, but not edit the transcript".

_EDITOR_RANK = ROLE_RANK["editor"]


def _resolve_folder(db: Session, folder: Optional[Folder | UUID]) -> Optional[Folder]:
    if folder is None or isinstance(folder, Folder):
        return folder
    return db.get(Folder, folder)


def can_act_on_video(
    db: Session, folder: Optional[Folder | UUID], principal: Principal
) -> bool:
    """retry/delete/move/fork of media: admin tier, developer, or editor rank.

    The developer grant (``LIBRARY_MEDIA_ROLES``) is library-scoped: it needs a
    folder the account can already see (``role_on_folder`` is not None), so an
    ungranted private folder grants nothing and an invisible unit is a 403.
    """
    folder = _resolve_folder(db, folder)
    if folder is None:
        return False
    role = role_on_folder(db, folder, principal)
    if role is None:
        return False
    return principal.role in LIBRARY_MEDIA_ROLES or ROLE_RANK.get(
        role, 0
    ) >= _EDITOR_RANK


def can_edit_transcript(
    db: Session, folder: Optional[Folder | UUID], principal: Principal
) -> bool:
    """edit/clean a transcript: admin tier, reviewer, or editor rank.

    The reviewer's row is ``edit/clean any``, so it is deliberately *not*
    conditioned on folder visibility — the transcript capability is orthogonal
    to the folder. The developer is excluded (``TRANSCRIPT_EDIT_ROLES`` does
    not hold them); only a genuine ``editor`` grant on the folder reaches this
    row for a non-tier account.
    """
    if principal.role in TRANSCRIPT_EDIT_ROLES:  # admin tier + reviewer
        return True
    folder = _resolve_folder(db, folder)
    if folder is None:
        return False
    role = role_on_folder(db, folder, principal)
    return role is not None and ROLE_RANK.get(role, 0) >= _EDITOR_RANK


def can_administer_folder(
    db: Session, folder: Optional[Folder | UUID], principal: Principal
) -> bool:
    """create/rename/delete/publish/share/zip/members: admin tier or editor.

    Neither library tier qualifies by role. A developer or reviewer reaches this
    row only through a genuine ``editor`` grant (owner or membership), the "by
    folder role" column of the matrix.
    """
    if principal.role in ADMIN_TIER:
        return True
    folder = _resolve_folder(db, folder)
    if folder is None:
        return False
    role = role_on_folder(db, folder, principal)
    return role is not None and ROLE_RANK.get(role, 0) >= _EDITOR_RANK


# ------------------------------------------------------------------ guards


def can_read_folder(db: Session, folder: Folder, principal: Principal) -> bool:
    return role_on_folder(db, folder, principal) is not None


def can_manage_folder(db: Session, folder: Folder, principal: Principal) -> bool:
    return can_administer_folder(db, folder, principal)


def require_read(db: Session, folder: Folder, principal: Principal) -> str:
    role = role_on_folder(db, folder, principal)
    if role is None:
        # Deliberately the same shape as a missing folder: whether a private
        # folder exists is not something an unauthorised caller should learn.
        raise AppError(
            404,
            "not_found",
            "Folder not found. It may be private, so sign in or unlock it with a passcode.",
            [{"id": str(folder.id)}],
        )
    return role


def require_manage(db: Session, folder: Folder, principal: Principal) -> str:
    role = role_on_folder(db, folder, principal)
    if not can_administer_folder(db, folder, principal):
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
    if principal.role not in OPERATOR_ROLES and not principal.is_bot_service:
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


def require_media_editor(db: Session, video: Video, principal: Principal) -> str:
    """A media action on one unit: metadata edit, delete, move, retry.

    Named for what it guards so a route that touches the media does not read
    like one that administers the folder. A `LIBRARY_MEDIA_ROLES` member
    (the developer) may act on a unit it can already see, whatever the role
    that made it visible — public, a viewer membership, ownership. This is the
    plan's "developer acts on any unit they can see" and it stops exactly
    there: an invisible unit still fails `require_video`'s 403, and folder
    administration keeps going through `require_manage`. Delegates to
    :func:`can_act_on_video` so the matrix lives in one place.
    """
    folder = db.get(Folder, video.folder_id) if video.folder_id else None
    role = video_role(db, video, principal)
    if role is not None and can_act_on_video(db, folder, principal):
        return role
    return require_video(db, video, principal, need="editor")


def require_media_folder(db: Session, folder: Folder, principal: Principal) -> str:
    """Write access to a destination folder for a media action (move/ingest).

    The destination's `require_manage` rule, kept under its own name for the
    same reason as `require_media_editor`, with the same library-tier widening:
    a `LIBRARY_MEDIA_ROLES` member whose destination is already visible may
    file media there. An invisible destination stays a 403, so a developer
    cannot push a unit into a private folder they hold no role on. Delegates to
    :func:`can_act_on_video`.
    """
    role = role_on_folder(db, folder, principal)
    if role is not None and can_act_on_video(db, folder, principal):
        return role
    return require_manage(db, folder, principal)


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
    the transcript text. Delegates to :func:`can_edit_transcript`, which also
    admits an editor-by-folder account and excludes the developer.
    """
    folder = db.get(Folder, video.folder_id) if video.folder_id else None
    if can_edit_transcript(db, folder, principal):
        return "editor"
    return require_video(db, video, principal, need="editor")
