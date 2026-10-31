"""The role engine: one module decides who may see what.

These tests are deliberately at the unit level. Every router above `access.py`
calls the same guards, so a bug here would silently widen or narrow the whole
dataset surface — worth pinning down directly.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select

from app.access import (
    ROLE_RANK,
    folder_chain,
    folder_or_404,
    require_comment,
    require_manage,
    require_read,
    require_video,
    role_on_folder,
    video_role,
    visible_folders,
)
from app.auth import Principal
from app.db import SessionLocal
from app.errors import AppError
from app.models import AdminUser, Folder, FolderMember, Video


@pytest.fixture
def user():
    """A dataset user (not the bootstrap operator)."""
    with SessionLocal() as db:
        account = AdminUser(email=f"user-{uuid4().hex[:8]}@test.local", role="user")
        db.add(account)
        db.commit()
        db.refresh(account)
        yield account
        db.delete(account)
        db.commit()


def _folder(name="Folder", visibility="private", parent=None, owner_id=None):
    with SessionLocal() as db:
        if owner_id is None:
            owner_id = db.execute(
                select(AdminUser).where(AdminUser.email == "admin@test.local")
            ).scalar_one().id
        folder = Folder(
            name=name,
            visibility=visibility,
            parent_id=parent,
            owner_id=owner_id,
        )
        db.add(folder)
        db.commit()
        db.refresh(folder)
        return folder.id


def _video(folder_id=None, created_by="super_admin:admin@test.local"):
    with SessionLocal() as db:
        video = Video(
            source_url="upload://access-test.mp4",
            platform="direct",
            folder_id=folder_id,
            created_by=created_by,
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        return video.id


def _as(db, model, id_):
    return db.get(model, id_)


def test_role_rank_orders_the_three_roles():
    assert ROLE_RANK["editor"] > ROLE_RANK["commenter"] > ROLE_RANK["viewer"]


# ------------------------------------------------------------------ principals


def test_admin_sees_every_folder_as_editor(user):
    fid = _folder()
    with SessionLocal() as db:
        folder = _as(db, Folder, fid)
        assert role_on_folder(db, folder, Principal("admin:a@b.c", "admin", "a@b.c")) == "editor"


def test_telegram_service_principal_is_an_editor(user):
    """The bot ingests on the operator's behalf, so it keeps full access."""
    fid = _folder()
    with SessionLocal() as db:
        folder = _as(db, Folder, fid)
        assert role_on_folder(db, folder, Principal("telegram:bot", "telegram")) == "editor"


def test_owner_is_an_editor_on_their_folder(user):
    fid = _folder(owner_id=user.id)
    with SessionLocal() as db:
        folder = _as(db, Folder, fid)
        principal = Principal(f"user:{user.email}", "user", user.email)
        assert role_on_folder(db, folder, principal) == "editor"


def test_user_gets_exactly_the_role_they_were_granted(user):
    fid = _folder()
    with SessionLocal() as db:
        folder = _as(db, Folder, fid)
        db.add(FolderMember(folder_id=fid, user_id=user.id, role="commenter"))
        db.commit()
        principal = Principal(f"user:{user.email}", "user", user.email)
        assert role_on_folder(db, folder, principal) == "commenter"


def test_public_folder_is_readable_by_an_anonymous_visitor():
    fid = _folder(visibility="public")
    with SessionLocal() as db:
        folder = _as(db, Folder, fid)
        anon = Principal("guest:anonymous", "guest")
        assert role_on_folder(db, folder, anon) == "viewer"


def test_private_folder_is_invisible_to_an_anonymous_visitor():
    fid = _folder(visibility="private")
    with SessionLocal() as db:
        folder = _as(db, Folder, fid)
        assert role_on_folder(db, folder, Principal("guest:anonymous", "guest")) is None


# -------------------------------------------------------------------- subtree


def test_access_inherits_down_the_folder_tree(user):
    """A user given a role on a parent folder may read its sub-folders (#6)."""
    parent = _folder(name="Parent")
    child = _folder(name="Child", parent=parent)
    with SessionLocal() as db:
        db.add(FolderMember(folder_id=parent, user_id=user.id, role="viewer"))
        db.commit()
        principal = Principal(f"user:{user.email}", "user", user.email)
        assert role_on_folder(db, _as(db, Folder, child), principal) == "viewer"


def test_a_public_ancestor_opens_the_whole_subtree():
    parent = _folder(name="Public parent", visibility="public")
    child = _folder(name="Child", parent=parent, visibility="private")
    with SessionLocal() as db:
        anon = Principal("guest:anonymous", "guest")
        assert role_on_folder(db, _as(db, Folder, child), anon) == "viewer"


def test_a_private_child_is_not_exposed_by_a_sibling(user):
    parent = _folder(name="Parent")
    _folder(name="Public child", parent=parent, visibility="public")
    hidden = _folder(name="Private child", parent=parent, visibility="private")
    with SessionLocal() as db:
        db.add(FolderMember(folder_id=parent, user_id=user.id, role="viewer"))
        db.commit()
        principal = Principal(f"user:{user.email}", "user", user.email)
        assert role_on_folder(db, _as(db, Folder, hidden), principal) == "viewer"
        # A stranger with no membership still cannot see it.
        assert role_on_folder(db, _as(db, Folder, hidden), Principal("guest:x", "guest")) is None


def test_folder_chain_goes_child_then_ancestors():
    grand = _folder(name="Grand")
    parent = _folder(name="Parent", parent=grand)
    child = _folder(name="Child", parent=parent)
    with SessionLocal() as db:
        chain = folder_chain(db, _as(db, Folder, child))
        assert [f.name for f in chain] == ["Child", "Parent", "Grand"]


def test_folder_chain_terminates_on_a_cycle():
    """A corrupt cycle (A -> B -> A) must not loop forever.

    The foreign key makes a *dangling* parent impossible, so the realistic
    corruption is a loop. `folder_chain` guards it with a visited set.
    """
    b = _folder(name="B")
    a = _folder(name="A", parent=b)
    with SessionLocal() as db:
        db.get(Folder, b).parent_id = a  # close the loop
        db.commit()
        chain = folder_chain(db, _as(db, Folder, a))
        assert [f.name for f in chain] == ["A", "B"]


# --------------------------------------------------------------- scoped tokens


def test_a_passcode_token_is_scoped_to_its_folder_only():
    opened = _folder(name="Opened")
    other = _folder(name="Other")
    with SessionLocal() as db:
        scoped = Principal(
            "guest:abc",
            "guest",
            None,
            frozenset({str(opened)}),
            {str(opened): "viewer"},
        )
        assert role_on_folder(db, _as(db, Folder, opened), scoped) == "viewer"
        assert role_on_folder(db, _as(db, Folder, other), scoped) is None


def test_a_scoped_editor_token_outranks_a_public_viewer():
    """An editor share link on a folder must not be downgraded by 'public'."""
    fid = _folder(name="Shared", visibility="public")
    with SessionLocal() as db:
        scoped = Principal(
            "guest:abc", "guest", None, frozenset({str(fid)}), {str(fid): "editor"}
        )
        assert role_on_folder(db, _as(db, Folder, fid), scoped) == "editor"


# ---------------------------------------------------------------------- videos


def test_the_ingesting_account_keeps_access_to_its_own_video(user):
    """A user who filed a video can still see it even outside any folder."""
    vid = _video(folder_id=None, created_by=f"user:{user.email}")
    with SessionLocal() as db:
        principal = Principal(f"user:{user.email}", "user", user.email)
        assert video_role(db, _as(db, Video, vid), principal) == "editor"


def test_a_video_inherits_its_folder_role(user):
    fid = _folder()
    vid = _video(folder_id=fid)
    with SessionLocal() as db:
        db.add(FolderMember(folder_id=fid, user_id=user.id, role="commenter"))
        db.commit()
        principal = Principal(f"user:{user.email}", "user", user.email)
        assert video_role(db, _as(db, Video, vid), principal) == "commenter"


def test_an_unfiled_video_is_private_to_its_creator():
    vid = _video(folder_id=None, created_by="user:someone@else.test")
    with SessionLocal() as db:
        stranger = Principal("user:me@test.local", "user", "me@test.local")
        assert video_role(db, _as(db, Video, vid), stranger) is None


# ------------------------------------------- a Telegram person is not the bot
#
# `telegram:bot` is the service identity; `telegram:<uid>` is a person acting
# through it. Both used to carry role "telegram", which granted editor
# everywhere — so any Telegram user who knew a UUID could read and overwrite
# another user's unit. These pin the narrowed rule: the person is an editor on
# their own ingests and inherits folder visibility like any ordinary account.


def test_a_telegram_person_is_an_editor_only_on_their_own_video():
    mine = _video(folder_id=None, created_by="telegram:777001")
    theirs = _video(folder_id=None, created_by="telegram:777002")
    with SessionLocal() as db:
        person = Principal("telegram:777001", "telegram", "777001")
        assert video_role(db, _as(db, Video, mine), person) == "editor"
        # The escalation regression: an unfiled unit owned by someone else is
        # invisible to a plain telegram:<uid>, exactly as to a plain account.
        assert video_role(db, _as(db, Video, theirs), person) is None


def test_a_telegram_person_sees_public_folders_but_not_private_ones():
    public = _folder(name="Public", visibility="public")
    private = _folder(name="Private", visibility="private")
    in_public = _video(folder_id=public, created_by="telegram:777002")
    in_private = _video(folder_id=private, created_by="telegram:777002")
    with SessionLocal() as db:
        person = Principal("telegram:777001", "telegram", "777001")
        assert video_role(db, _as(db, Video, in_public), person) == "viewer"
        assert video_role(db, _as(db, Video, in_private), person) is None
        # The folder tree agrees: only the public folder is listed.
        assert private not in visible_folders(db, person)
        assert public in visible_folders(db, person)


def test_the_bot_service_identity_keeps_its_blanket_editor_grant():
    private = _folder(name="Private", visibility="private")
    unfiled = _video(folder_id=None, created_by="telegram:777002")
    with SessionLocal() as db:
        bot = Principal("telegram:bot", "telegram")
        assert bot.is_bot_service is True
        assert role_on_folder(db, _as(db, Folder, private), bot) == "editor"
        assert video_role(db, _as(db, Video, unfiled), bot) == "editor"
        # And the person form is never mistaken for the service one.
        person = Principal("telegram:777001", "telegram", "777001")
        assert person.is_bot_service is False


# ---------------------------------------------------------------------- guards


def test_require_read_reports_a_private_folder_as_missing(user):
    """Whether a private folder exists is not for an outsider to learn."""
    fid = _folder(visibility="private")
    with SessionLocal() as db:
        folder = _as(db, Folder, fid)
        with pytest.raises(AppError) as denied:
            require_read(db, folder, Principal("guest:anonymous", "guest"))
        assert denied.value.status_code == 404
        assert denied.value.code == "not_found"


def test_require_read_accepts_any_role():
    fid = _folder(visibility="public")
    with SessionLocal() as db:
        assert require_read(db, _as(db, Folder, fid), Principal("guest:a", "guest")) == "viewer"


def test_require_manage_rejects_a_viewer():
    fid = _folder(visibility="public")
    with SessionLocal() as db:
        with pytest.raises(AppError) as denied:
            require_manage(db, _as(db, Folder, fid), Principal("guest:a", "guest"))
        assert denied.value.status_code == 403
        assert denied.value.code == "forbidden"


def test_require_comment_rejects_a_viewer_but_accepts_a_commenter(user):
    fid = _folder()
    with SessionLocal() as db:
        db.add(FolderMember(folder_id=fid, user_id=user.id, role="commenter"))
        db.commit()
        viewer = Principal("guest:a", "guest")
        with pytest.raises(AppError):
            require_comment(db, _as(db, Folder, fid), viewer)

        commenter = Principal(f"user:{user.email}", "user", user.email)
        assert require_comment(db, _as(db, Folder, fid), commenter) == "commenter"


def test_require_video_enforces_the_needed_role(user):
    fid = _folder()
    vid = _video(folder_id=fid)
    with SessionLocal() as db:
        db.add(FolderMember(folder_id=fid, user_id=user.id, role="viewer"))
        db.commit()
        principal = Principal(f"user:{user.email}", "user", user.email)
        assert require_video(db, _as(db, Video, vid), principal, "viewer") == "viewer"
        with pytest.raises(AppError) as denied:
            require_video(db, _as(db, Video, vid), principal, "commenter")
        assert denied.value.status_code == 403
        assert denied.value.details[0]["required"] == "commenter"


def test_require_video_denies_a_stranger_outright():
    vid = _video(folder_id=None, created_by="user:someone@else.test")
    with SessionLocal() as db:
        stranger = Principal("user:me@test.local", "user", "me@test.local")
        with pytest.raises(AppError) as denied:
            require_video(db, _as(db, Video, vid), stranger)
        assert denied.value.status_code == 403


# ------------------------------------------------------------ visible_folders


def test_visible_folders_hides_private_folders_from_strangers(user):
    mine = _folder(name="Mine")
    _folder(name="Not mine", visibility="private")
    _folder(name="Everyone", visibility="public")
    with SessionLocal() as db:
        db.add(FolderMember(folder_id=mine, user_id=user.id, role="viewer"))
        db.commit()
        principal = Principal(f"user:{user.email}", "user", user.email)
        visible = visible_folders(db, principal)
        names = {db.get(Folder, fid).name for fid in visible}
        assert names == {"Mine", "Everyone"}


def test_visible_folders_includes_a_folder_role_for_the_caller(user):
    mine = _folder(name="Mine")
    with SessionLocal() as db:
        db.add(FolderMember(folder_id=mine, user_id=user.id, role="commenter"))
        db.commit()
        principal = Principal(f"user:{user.email}", "user", user.email)
        assert visible_folders(db, principal)[mine] == "commenter"


def test_visible_folders_for_admin_is_everything():
    a = _folder(name="A", visibility="private")
    b = _folder(name="B", visibility="public")
    with SessionLocal() as db:
        visible = visible_folders(db, Principal("admin:a@b.c", "admin", "a@b.c"))
        assert visible == {a: "editor", b: "editor"}


def test_folder_or_404_is_a_uniform_404():
    with SessionLocal() as db:
        with pytest.raises(AppError) as missing:
            folder_or_404(db, uuid4())
        assert missing.value.status_code == 404
        assert missing.value.details[0]["id"]
