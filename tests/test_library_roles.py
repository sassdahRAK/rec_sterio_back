"""F2: the library tier's floor on every folder (contract §0 role matrix).

`role_on_folder` used to resolve a `developer` or `reviewer` with no explicit
per-folder grant as a plain viewer or as nothing at all. The matrix says the
developer is the *library operator* — read, edit transcripts, delete, move and
fork anywhere — and the reviewer reads everywhere and edits any transcript but
holds no destructive right. These tests drive the real resolution helpers and
the real guards (`require_read`, `require_transcript_editor`,
`require_media_editor`, `require_media_folder`, `require_operator`), not mocks,
so a drift in `access.py` fails here instead of in a router.

The critical non-regression is the ordinary `user`: the tier widening must not
leak into their resolution. A `user`, a guest, a passcode/share session and the
Telegram service keep exactly the rules they had.
"""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from app.access import (
    require_media_editor,
    require_media_folder,
    require_operator,
    require_read,
    require_transcript_editor,
    require_video,
    role_on_folder,
    video_role,
    visible_folders,
)
from app.auth import Principal
from app.db import SessionLocal
from app.errors import AppError
from app.models import AdminUser, Folder, FolderMember, Video

ADMIN_EMAIL = "admin@test.local"

# The six actors §0's matrix covers. `guest` is the anonymous visitor in the
# public/private rows and a passcode-scoped session in the grant row.
ACTORS = ("super_admin", "admin", "developer", "reviewer", "user", "guest")
SCENARIOS = ("public", "private", "grant")
CAPABILITIES = ("read", "edit_transcript", "delete_video", "move_fork", "retry")

READ_PUBLIC = {"read": True, "edit_transcript": False, "delete_video": False,
               "move_fork": False, "retry": False}
READ_ALL = {"read": True, "edit_transcript": True, "delete_video": False,
            "move_fork": False, "retry": False}
EDITOR_NO_OPERATOR = {"read": True, "edit_transcript": True, "delete_video": True,
                      "move_fork": True, "retry": False}
FULL = {"read": True, "edit_transcript": True, "delete_video": True,
        "move_fork": True, "retry": True}
NOTHING = dict.fromkeys(CAPABILITIES, False)

# role -> scenario -> capability -> allowed
EXPECTED = {
    "super_admin": {"public": FULL, "private": FULL, "grant": FULL},
    "admin": {"public": FULL, "private": FULL, "grant": FULL},
    # The library tier: editor everywhere, including a private folder nobody
    # granted them (the point of F2). A weak explicit grant cannot lower it.
    "developer": {"public": FULL, "private": FULL, "grant": FULL},
    # Read + transcript edit everywhere; no destructive right unless an
    # explicit "editor" grant raises them on that one folder. `retry` also
    # passes `require_operator`, which the reviewer never holds.
    "reviewer": {"public": READ_ALL, "private": READ_ALL, "grant": EDITOR_NO_OPERATOR},
    # Ordinary member: public read only, nothing on a private folder they hold
    # no role on, and exactly their grant where they hold one.
    "user": {"public": READ_PUBLIC, "private": NOTHING, "grant": EDITOR_NO_OPERATOR},
    "guest": {"public": READ_PUBLIC, "private": NOTHING, "grant": EDITOR_NO_OPERATOR},
}


# ------------------------------------------------------------------ fixtures


def _seed_admin_id() -> UUID:
    with SessionLocal() as db:
        return db.execute(
            select(AdminUser).where(AdminUser.email == ADMIN_EMAIL)
        ).scalar_one().id


def _make_account(role: str) -> tuple[UUID, str]:
    email = f"{role}-{uuid4().hex[:8]}@test.local"
    with SessionLocal() as db:
        account = AdminUser(email=email, role=role)
        db.add(account)
        db.commit()
        db.refresh(account)
        return account.id, email


def _make_folder(owner_id: UUID, visibility: str, name: str) -> UUID:
    with SessionLocal() as db:
        folder = Folder(name=name, visibility=visibility, owner_id=owner_id)
        db.add(folder)
        db.commit()
        db.refresh(folder)
        return folder.id


def _make_video(folder_id: UUID) -> UUID:
    with SessionLocal() as db:
        video = Video(
            source_url=f"upload://{uuid4().hex}.mp4",
            platform="direct",
            folder_id=folder_id,
            created_by=f"super_admin:{ADMIN_EMAIL}",
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        return video.id


def _grant(folder_id: UUID, user_id: UUID, role: str) -> None:
    with SessionLocal() as db:
        db.add(FolderMember(folder_id=folder_id, user_id=user_id, role=role))
        db.commit()


def _build(actor: str, scenario: str):
    """One table cell: a folder owned by the seeded operator, a video in it,
    and the principal the actor presents. The video is never theirs, so the
    creator shortcut in `video_role` cannot mask the folder rule."""
    owner_id = _seed_admin_id()
    account_id = account_email = None
    if actor in ("developer", "reviewer", "user"):
        account_id, account_email = _make_account(actor)

    visibility = "public" if scenario == "public" else "private"
    folder_id = _make_folder(owner_id, visibility, f"{actor}-{scenario}")
    video_id = _make_video(folder_id)

    if scenario == "grant" and account_id is not None:
        # reviewer/user are raised to editor; the developer is given a *weaker*
        # grant on purpose so the test proves the tier is not downgraded.
        _grant(folder_id, account_id, "viewer" if actor == "developer" else "editor")

    if actor == "super_admin":
        principal = Principal(f"super_admin:{ADMIN_EMAIL}", "super_admin", ADMIN_EMAIL)
    elif actor == "admin":
        principal = Principal("admin:second@test.local", "admin", "second@test.local")
    elif actor == "guest":
        if scenario == "grant":
            principal = Principal(
                "guest:scoped",
                "guest",
                None,
                frozenset({str(folder_id)}),
                {str(folder_id): "editor"},
            )
        else:
            principal = Principal("guest:anonymous", "guest")
    else:
        principal = Principal(f"{actor}:{account_email}", actor, account_email)

    return folder_id, video_id, principal


def _can(db, capability: str, folder: Folder, video: Video, principal: Principal) -> bool:
    """Run the real guard the route runs for this capability."""
    try:
        if capability == "read":
            require_read(db, folder, principal)
        elif capability == "edit_transcript":
            require_transcript_editor(db, video, principal)
        elif capability == "delete_video":
            require_media_editor(db, video, principal)
        elif capability == "move_fork":
            # Moving/forking *into* the folder: the destination needs manage.
            require_media_folder(db, folder, principal)
        elif capability == "retry":
            # Video retry needs media edit on the folder; job retry is the
            # operator surface. Both must pass for "retry" to be allowed.
            require_media_editor(db, video, principal)
            require_operator(principal)
        else:  # pragma: no cover - guards the table itself
            raise AssertionError(capability)
    except AppError:
        return False
    return True


def _capabilities(folder_id: UUID, video_id: UUID, principal: Principal) -> dict:
    with SessionLocal() as db:
        folder = db.get(Folder, folder_id)
        video = db.get(Video, video_id)
        return {
            capability: _can(db, capability, folder, video, principal)
            for capability in CAPABILITIES
        }


def _role(folder_id: UUID, principal: Principal) -> str | None:
    with SessionLocal() as db:
        return role_on_folder(db, db.get(Folder, folder_id), principal)


# --------------------------------------------------------------- the matrix


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("actor", ACTORS)
def test_capability_matrix(actor: str, scenario: str):
    folder_id, video_id, principal = _build(actor, scenario)
    assert _capabilities(folder_id, video_id, principal) == EXPECTED[actor][scenario]


# ------------------------------------------------------- pinned negative cases


def test_a_developer_is_an_editor_on_a_private_folder_no_one_granted():
    """The point of the library tier: destructive rights without a grant."""
    folder_id, video_id, principal = _build("developer", "private")
    assert _role(folder_id, principal) == "editor"
    caps = _capabilities(folder_id, video_id, principal)
    assert caps["delete_video"] is True
    assert caps["move_fork"] is True
    assert caps["retry"] is True
    with SessionLocal() as db:
        assert video_role(db, db.get(Video, video_id), principal) == "editor"


def test_a_reviewer_has_no_destructive_right_without_an_editor_grant():
    """Public and private, granted nothing: read + transcript only."""
    for scenario in ("public", "private"):
        folder_id, video_id, principal = _build("reviewer", scenario)
        caps = _capabilities(folder_id, video_id, principal)
        assert caps["read"] is True, scenario
        assert caps["edit_transcript"] is True, scenario
        assert caps["delete_video"] is False, scenario
        assert caps["move_fork"] is False, scenario
        assert caps["retry"] is False, scenario


def test_a_reviewer_reads_a_private_folder_but_the_media_stays_closed():
    """The reviewer's library reach is read-only on a private folder they
    were never granted."""
    folder_id, video_id, principal = _build("reviewer", "private")
    assert _role(folder_id, principal) == "viewer"
    with SessionLocal() as db:
        assert video_role(db, db.get(Video, video_id), principal) == "viewer"


def test_an_editor_grant_raises_the_reviewer_but_never_the_job_queue():
    """§0: an explicit editor grant still wins (it grants more), but the
    operator surface is a role, not a folder, so retry jobs stays closed."""
    folder_id, video_id, principal = _build("reviewer", "grant")
    assert _role(folder_id, principal) == "editor"
    caps = _capabilities(folder_id, video_id, principal)
    assert caps["delete_video"] is True
    assert caps["move_fork"] is True
    assert caps["retry"] is False


def test_a_viewer_grant_does_not_downgrade_the_developer():
    folder_id, _, principal = _build("developer", "grant")
    assert _role(folder_id, principal) == "editor"
    with SessionLocal() as db:
        assert visible_folders(db, principal)[folder_id] == "editor"


def test_an_ordinary_user_is_still_denied_on_a_private_folder_they_hold_no_role_on():
    """The regression that matters most: the tier widening must not touch
    `user`. A private folder with no membership stays invisible (404, not 403),
    and no capability resolves."""
    folder_id, video_id, principal = _build("user", "private")
    assert _role(folder_id, principal) is None
    assert _capabilities(folder_id, video_id, principal) == NOTHING
    with SessionLocal() as db:
        with pytest.raises(AppError) as denied:
            require_read(db, db.get(Folder, folder_id), principal)
    assert denied.value.status_code == 404


def test_a_guest_is_still_denied_on_a_private_folder():
    folder_id, video_id, principal = _build("guest", "private")
    assert _role(folder_id, principal) is None
    assert _capabilities(folder_id, video_id, principal) == NOTHING
    with SessionLocal() as db:
        # The bot is unchanged too: a service principal sees every folder.
        assert role_on_folder(
            db, db.get(Folder, folder_id), Principal("telegram:bot", "telegram")
        ) == "editor"


def test_a_passcode_session_stays_scoped_to_its_one_folder():
    """A guest token is unchanged: its role holds on the folder it was issued
    for and nowhere else."""
    scoped = _make_folder(_seed_admin_id(), "private", "Opened")
    other = _make_folder(_seed_admin_id(), "private", "Other")
    principal = Principal(
        "guest:passcode",
        "guest",
        None,
        frozenset({str(scoped)}),
        {str(scoped): "viewer"},
    )
    assert _role(scoped, principal) == "viewer"
    assert _role(other, principal) is None


def test_the_admin_tier_and_the_bot_keep_their_resolution():
    private = _make_folder(_seed_admin_id(), "private", "Untouched")
    for principal in (
        Principal(f"super_admin:{ADMIN_EMAIL}", "super_admin", ADMIN_EMAIL),
        Principal("admin:second@test.local", "admin", "second@test.local"),
        Principal("telegram:bot", "telegram"),
    ):
        assert _role(private, principal) == "editor", principal.role
        require_operator(principal)


# ------------------------------------------------- the listing's role source


def test_the_library_tier_sees_the_whole_folder_tree():
    """`visible_folders` is what the folder listing and the listings' scope
    read. The developer sees every folder as editor; the reviewer as a viewer,
    raised to editor only where they hold an explicit editor grant. A plain
    user sees the public folder and their own grant, never the private one."""
    owner = _seed_admin_id()
    public = _make_folder(owner, "public", "Public")
    private = _make_folder(owner, "private", "Vault")
    granted = _make_folder(owner, "private", "Granted")

    reviewer_id, reviewer_email = _make_account("reviewer")
    _grant(granted, reviewer_id, "editor")
    reviewer = Principal(f"reviewer:{reviewer_email}", "reviewer", reviewer_email)

    developer_id, developer_email = _make_account("developer")
    developer = Principal(f"developer:{developer_email}", "developer", developer_email)

    user_id, user_email = _make_account("user")
    _grant(granted, user_id, "viewer")
    user = Principal(f"user:{user_email}", "user", user_email)

    with SessionLocal() as db:
        dev_view = visible_folders(db, developer)
        rev_view = visible_folders(db, reviewer)
        user_view = visible_folders(db, user)

    assert dev_view == {public: "editor", private: "editor", granted: "editor"}
    assert rev_view == {public: "viewer", private: "viewer", granted: "editor"}
    # Unchanged member resolution: private is absent, not merely lower.
    assert user_view == {public: "viewer", granted: "viewer"}
