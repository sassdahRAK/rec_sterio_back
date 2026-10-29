"""The library tier widens media *actions*, never folder visibility (§0).

`developer` and `reviewer` hold no implicit role on any folder. A private
folder with no grant is invisible to them — `role_on_folder` is None, the
folder is absent from `visible_folders`, and `require_read` answers 404 —
exactly as for a plain `user` or a guest. What the tier actually buys is
per-capability:

* `require_media_editor` / `require_media_folder` let a `LIBRARY_MEDIA_ROLES`
  member (the developer) delete/move/retry/fork the media of a unit it can
  already see, even when the role that made it visible is only `viewer`. An
  invisible unit still 403s, so there is nothing there to act on.
* `require_transcript_editor` lets the reviewer edit any transcript
  (`TRANSCRIPT_EDIT_ROLES`) — a transcript capability, orthogonal to whether
  the folder is visible.

Folder administration is untouched: `require_manage` still needs `editor` on
that folder, which the library tier only holds as owner or with an explicit
editor grant. These tests drive the real resolution helpers and the real
guards (`require_read`, `require_manage`, `require_transcript_editor`,
`require_media_editor`, `require_media_folder`, `require_operator`), not
mocks, so a drift in `access.py` fails here instead of in a router.

The critical non-regression is the ordinary `user`: nothing widens for them. A
`user`, a guest, a passcode/share session and the Telegram service keep
exactly the rules they had.
"""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from app.access import (
    require_manage,
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
# The reviewer on an invisible private folder: the folder itself is a 404, but
# transcript editing is a global capability (`TRANSCRIPT_EDIT_ROLES`), so that
# one column still passes.
TRANSCRIPT_ONLY = {"read": False, "edit_transcript": True, "delete_video": False,
                   "move_fork": False, "retry": False}
EDITOR_NO_OPERATOR = {"read": True, "edit_transcript": True, "delete_video": True,
                      "move_fork": True, "retry": False}
# The developer on a visible unit: the media widening applies, but transcript
# editing is a separate role (`TRANSCRIPT_EDIT_ROLES`) they do not hold.
LIBRARY_MEDIA = {"read": True, "edit_transcript": False, "delete_video": True,
                 "move_fork": True, "retry": True}
FULL = {"read": True, "edit_transcript": True, "delete_video": True,
        "move_fork": True, "retry": True}
NOTHING = dict.fromkeys(CAPABILITIES, False)

# role -> scenario -> capability -> allowed
EXPECTED = {
    "super_admin": {"public": FULL, "private": FULL, "grant": FULL},
    "admin": {"public": FULL, "private": FULL, "grant": FULL},
    # The library tier does NOT widen visibility. A developer is a viewer on a
    # public folder, nothing at all on an ungranted private one (404, not
    # 403), and a viewer where they hold a viewer grant. The media widening
    # (`require_media_editor`/`require_media_folder`) is what lets them
    # delete/move/retry/fork/ingest — only where the unit is already visible;
    # `retry` also passes `require_operator`, which the developer holds.
    "developer": {"public": LIBRARY_MEDIA, "private": NOTHING, "grant": LIBRARY_MEDIA},
    # Visible: read + transcript edit, no destructive right unless an explicit
    # "editor" grant raises them on that one folder. Invisible: the transcript
    # widening survives (it is a role, not a folder), nothing else does.
    "reviewer": {"public": READ_ALL, "private": TRANSCRIPT_ONLY, "grant": EDITOR_NO_OPERATOR},
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
        # A real grant: viewer for the developer (the media widening must work
        # at viewer), editor for the reviewer/user (the ordinary role rules).
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


def test_a_developer_cannot_manage_an_ungranted_private_folder():
    """The regression: `require_manage` must not pass on a private folder the
    developer holds no role on. It is a 403 and nothing is renamed/deleted."""
    folder_id, _, principal = _build("developer", "private")
    with SessionLocal() as db:
        with pytest.raises(AppError) as denied:
            require_manage(db, db.get(Folder, folder_id), principal)
    assert denied.value.status_code == 403
    assert denied.value.details[0]["role"] is None


def test_a_developer_cannot_see_an_ungranted_private_folder():
    """Invisible means invisible: no role, absent from the listing, 404 on read
    — the same shape a plain user gets, so existence is not disclosed."""
    folder_id, video_id, principal = _build("developer", "private")
    assert _role(folder_id, principal) is None
    with SessionLocal() as db:
        assert folder_id not in visible_folders(db, principal)
        assert video_role(db, db.get(Video, video_id), principal) is None
        with pytest.raises(AppError) as denied:
            require_read(db, db.get(Folder, folder_id), principal)
    assert denied.value.status_code == 404


def test_a_developer_cannot_move_a_video_into_an_ungranted_private_folder():
    """`require_media_folder` widens only for a visible destination. Here the
    destination is invisible, so a move/ingest/fork into it is a 403 — and so
    is reading the unit already inside it."""
    folder_id, video_id, principal = _build("developer", "private")
    with SessionLocal() as db:
        with pytest.raises(AppError) as denied:
            require_media_folder(db, db.get(Folder, folder_id), principal)
        assert denied.value.status_code == 403
        with pytest.raises(AppError) as denied_video:
            require_media_editor(db, db.get(Video, video_id), principal)
        assert denied_video.value.status_code == 403


def test_a_developer_still_acts_on_media_in_a_folder_they_can_see():
    """The widening itself: at `viewer` (public, or a viewer grant) the
    developer deletes/moves/retries the media, but the very same folder is
    still closed to `require_manage` — media actions are not administration."""
    for scenario in ("public", "grant"):
        folder_id, video_id, principal = _build("developer", scenario)
        with SessionLocal() as db:
            folder = db.get(Folder, folder_id)
            video = db.get(Video, video_id)
            assert role_on_folder(db, folder, principal) == "viewer", scenario
            assert video_role(db, video, principal) == "viewer", scenario
            assert require_media_editor(db, video, principal) == "viewer", scenario
            assert require_media_folder(db, folder, principal) == "viewer", scenario
            require_operator(principal)
            with pytest.raises(AppError) as denied:
                require_manage(db, folder, principal)
            assert denied.value.status_code == 403, scenario


def test_a_reviewer_has_no_destructive_right_without_an_editor_grant():
    """Public: read + transcript only. Private and ungranted: invisible, so
    only the global transcript widening still resolves."""
    public_id, public_video, public_principal = _build("reviewer", "public")
    caps = _capabilities(public_id, public_video, public_principal)
    assert caps == READ_ALL
    for capability in ("delete_video", "move_fork", "retry"):
        assert caps[capability] is False, capability

    private_id, private_video, private_principal = _build("reviewer", "private")
    caps = _capabilities(private_id, private_video, private_principal)
    assert caps == TRANSCRIPT_ONLY


def test_a_reviewer_cannot_read_a_private_folder_they_hold_no_role_on():
    """The reviewer's library reach is *not* read-everywhere: an ungranted
    private folder is invisible. Transcript editing still works because it is
    a role capability, not a folder grant."""
    folder_id, video_id, principal = _build("reviewer", "private")
    assert _role(folder_id, principal) is None
    with SessionLocal() as db:
        assert folder_id not in visible_folders(db, principal)
        with pytest.raises(AppError) as denied:
            require_read(db, db.get(Folder, folder_id), principal)
        assert denied.value.status_code == 404
        # The transcript route is deliberately orthogonal to visibility.
        require_transcript_editor(db, db.get(Video, video_id), principal)


def test_an_editor_grant_raises_the_reviewer_but_never_the_job_queue():
    """§0: an explicit editor grant still wins (it grants more), but the
    operator surface is a role, not a folder, so retry jobs stays closed."""
    folder_id, video_id, principal = _build("reviewer", "grant")
    assert _role(folder_id, principal) == "editor"
    caps = _capabilities(folder_id, video_id, principal)
    assert caps["delete_video"] is True
    assert caps["move_fork"] is True
    assert caps["retry"] is False


def test_a_viewer_grant_gives_the_developer_viewer_not_editor():
    """A grant is respected as written: viewer stays viewer on the folder, so
    folder administration stays closed. Only the media widening applies."""
    folder_id, video_id, principal = _build("developer", "grant")
    assert _role(folder_id, principal) == "viewer"
    with SessionLocal() as db:
        assert visible_folders(db, principal)[folder_id] == "viewer"
        with pytest.raises(AppError) as denied:
            require_manage(db, db.get(Folder, folder_id), principal)
    assert denied.value.status_code == 403
    caps = _capabilities(folder_id, video_id, principal)
    assert caps["read"] is True
    assert caps["delete_video"] is True
    assert caps["move_fork"] is True
    assert caps["retry"] is True


def test_an_ordinary_user_is_still_denied_on_a_private_folder_they_hold_no_role_on():
    """The regression that matters most: nothing widens for `user`. A private
    folder with no membership stays invisible (404, not 403), and no
    capability resolves."""
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


def test_the_library_tier_does_not_widen_the_folder_tree():
    """`visible_folders` is what the folder listing and the listings' scope
    read. The developer gets public folders at viewer plus their grants —
    never the ungranted private folder. The reviewer is the same, raised to
    editor only where they hold an explicit editor grant. A plain user sees
    the public folder and their own grant, never the private one."""
    owner = _seed_admin_id()
    public = _make_folder(owner, "public", "Public")
    private = _make_folder(owner, "private", "Vault")
    granted = _make_folder(owner, "private", "Granted")

    reviewer_id, reviewer_email = _make_account("reviewer")
    _grant(granted, reviewer_id, "editor")
    reviewer = Principal(f"reviewer:{reviewer_email}", "reviewer", reviewer_email)

    developer_id, developer_email = _make_account("developer")
    _grant(granted, developer_id, "viewer")
    developer = Principal(f"developer:{developer_email}", "developer", developer_email)

    user_id, user_email = _make_account("user")
    _grant(granted, user_id, "viewer")
    user = Principal(f"user:{user_email}", "user", user_email)

    with SessionLocal() as db:
        dev_view = visible_folders(db, developer)
        rev_view = visible_folders(db, reviewer)
        user_view = visible_folders(db, user)

    # The ungranted private folder is absent for the library tier too.
    assert dev_view == {public: "viewer", granted: "viewer"}
    assert rev_view == {public: "viewer", granted: "editor"}
    # Unchanged member resolution: private is absent, not merely lower.
    assert user_view == {public: "viewer", granted: "viewer"}
