"""Bot user routes: per-user consent records and cookies.txt storage.

The bot forwards a user's action with the shared service token plus an
`X-Requested-By: telegram:<id>` header. Both are required here, so one
user's cookies can never be read, replaced or withdrawn through another
user's request. The cookies files themselves live on disk (one 0600 file
per user id); only their checksum is stored in the database.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import Principal
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import AdminUser, BotUser, LinkCode
from app.roles import ADMIN_TIER
from app.routers.passcodes import hash_code
from app.security import detect_platform
from app.services import storage

logger = logging.getLogger("recsterio.bot_users")

router = APIRouter(prefix="/bot-users", tags=["bot-users"])

# The consent text the bot showed the user. Bumping this invalidates every
# earlier consent: the user must run /cookies again.
CONSENT_VERSION = "2026-10-28.1"

# Platforms that refuse an anonymous server, so the download must run under
# the requesting user's own session (their uploaded cookies).
SESSION_GATED_PLATFORMS = ("tiktok", "facebook")

REQUESTED_BY_HEADER = "X-Requested-By"


def _require_bot_user(request: Request, telegram_id: int) -> None:
    """Self-contained credential check for a bot user's own record.

    Requires BOTH the shared bot service token and an `X-Requested-By`
    header naming the same numeric id the path addresses.
    """
    _require_bot_service(request)
    requested_by = request.headers.get(REQUESTED_BY_HEADER)
    if requested_by != f"telegram:{telegram_id}":
        raise AppError(
            403,
            "forbidden",
            "This request must be made by the Telegram user it acts for.",
            [{"header": REQUESTED_BY_HEADER}],
        )


def _require_bot_service(request: Request) -> None:
    """The service token alone: routes the bot calls with no user attached.

    The `X-Requested-By` header is what turns the service identity into a
    user's own, so it is deliberately NOT required for the service-level
    routes (the admin-id lookup answers about accounts, not about a user).
    """
    auth = request.headers.get("authorization") or ""
    token = settings.bot_api_token
    if not token or auth != f"Bearer {token}":
        raise AppError(
            401, "not_authenticated", "A valid bot service token is required."
        )


def _get_row(db: Session, telegram_id: int) -> Optional[BotUser]:
    return db.execute(
        select(BotUser).where(BotUser.telegram_user_id == telegram_id)
    ).scalar_one_or_none()


class BotUserOut(BaseModel):
    telegram_id: int
    consented: bool
    consented_at: Optional[datetime] = None
    consent_version: Optional[str] = None
    cookies_present: bool
    cookies_updated_at: Optional[datetime] = None
    # The console account this Telegram user is linked to (None when unlinked),
    # so the bot can say whose dataset a /link actually opened.
    linked_email: Optional[str] = None
    linked_name: Optional[str] = None
    linked_role: Optional[str] = None


class ConsentIn(BaseModel):
    consent_version: str


class CookiesOut(BaseModel):
    cookies_present: bool
    cookies_updated_at: Optional[datetime] = None


class AdminIdsOut(BaseModel):
    """Telegram ids that reach an active console admin through the bot."""

    ids: List[int] = Field(default_factory=list)


class BotLinkIn(BaseModel):
    """A link code the console showed the user, plus the two-phase flag.

    `confirm=False` is the dry run ("this code will link you to <email>");
    `confirm=True` redeems it.
    """

    code: str
    confirm: bool


class BotLinkPreviewOut(BaseModel):
    account_email: str


class BotLinkResultOut(BaseModel):
    account_email: str
    account_name: Optional[str] = None
    account_role: str


def _linked_account(db: Session, telegram_id: int) -> Optional[AdminUser]:
    return db.execute(
        select(AdminUser).where(AdminUser.telegram_user_id == telegram_id)
    ).scalar_one_or_none()


def _user_out(db: Session, telegram_id: int, row: Optional[BotUser]) -> BotUserOut:
    linked = _linked_account(db, telegram_id)
    return BotUserOut(
        telegram_id=telegram_id,
        consented=bool(row.consented) if row is not None else False,
        consented_at=row.consented_at if row is not None else None,
        consent_version=row.consent_version if row is not None else None,
        cookies_present=storage.cookie_file(telegram_id).exists(),
        cookies_updated_at=row.cookies_updated_at if row is not None else None,
        linked_email=linked.email if linked is not None else None,
        linked_name=linked.name if linked is not None else None,
        linked_role=linked.role if linked is not None else None,
    )


# A STATIC path must be declared before /{telegram_id}: FastAPI matches in
# registration order, and "admin-ids" would otherwise be parsed as a telegram
# id and rejected as a validation error.
@router.get("/admin-ids", response_model=AdminIdsOut)
def list_admin_ids(
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_bot_service),
) -> AdminIdsOut:
    """Telegram ids the bot may treat as a console admin.

    The service token alone is enough: this answers about accounts, not about
    any one user's record, so no `X-Requested-By` applies.
    """
    ids = (
        db.execute(
            select(AdminUser.telegram_user_id)
            .where(
                AdminUser.telegram_user_id.is_not(None),
                AdminUser.role.in_(ADMIN_TIER),
                AdminUser.is_active.is_(True),
            )
            .order_by(AdminUser.telegram_user_id)
        )
        .scalars()
        .all()
    )
    return AdminIdsOut(ids=[int(i) for i in ids])


@router.get("/{telegram_id}", response_model=BotUserOut)
def get_bot_user(
    telegram_id: int,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_bot_user),
) -> BotUserOut:
    row = _get_row(db, telegram_id)
    if row is None:
        raise AppError(
            404, "not_found", "No bot user with this id.", [{"telegram_id": telegram_id}]
        )
    return _user_out(db, telegram_id, row)


@router.post("/{telegram_id}/consent", response_model=BotUserOut)
def give_consent(
    telegram_id: int,
    payload: ConsentIn,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_bot_user),
) -> BotUserOut:
    if payload.consent_version != CONSENT_VERSION:
        raise AppError(
            422,
            "version_mismatch",
            "The consent text has changed — run /cookies again.",
            [{"field": "consent_version", "expected": CONSENT_VERSION}],
        )
    now = datetime.now(timezone.utc)
    row = _get_row(db, telegram_id)
    if row is None:
        row = BotUser(telegram_user_id=telegram_id)
        db.add(row)
    row.consented = True
    row.consented_at = now
    row.consent_version = CONSENT_VERSION
    db.commit()
    db.refresh(row)
    return _user_out(db, telegram_id, row)


def _find_live_link_code(db: Session, code: str) -> LinkCode:
    """Look up a link code by hash: unrevoked and unexpired, or nothing.

    Every failure is the same 401 so a wrong, an expired and an already-used
    code are indistinguishable — the passcode precedent — and the caller
    learns nothing about which codes exist.
    """
    now = datetime.now(timezone.utc)
    link_code = db.execute(
        select(LinkCode).where(
            LinkCode.revoked_at.is_(None), LinkCode.code_hash == hash_code(code)
        )
    ).scalar_one_or_none()
    if link_code is None or (
        link_code.expires_at is not None and link_code.expires_at <= now
    ):
        raise AppError(401, "unauthorized", "That link code is not valid.")
    return link_code


@router.post("/{telegram_id}/link", response_model=None)
def link_account(
    telegram_id: int,
    payload: BotLinkIn,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_bot_user),
) -> BotLinkPreviewOut | BotLinkResultOut:
    """Bind this Telegram user to the console account behind a link code.

    Two-phase: `confirm=false` is a dry run that names the account so the bot
    can ask "link to <email>?" without consuming the code; `confirm=true`
    redeems it — exactly once, because redeeming revokes the code. The two
    phases answer different bodies, so no single response_model applies.
    """
    link_code = _find_live_link_code(db, payload.code)
    admin_user = db.get(AdminUser, link_code.admin_user_id)
    if admin_user is None:
        # The account was deleted while its code was outstanding. To the
        # caller this is indistinguishable from any other dead code.
        raise AppError(401, "unauthorized", "That link code is not valid.")

    if not payload.confirm:
        # Dry run: name the account, redeem nothing.
        return BotLinkPreviewOut(account_email=admin_user.email)

    # Conflict checks, both directions of the one-to-one pairing:
    # (a) this Telegram account already belongs to a different console
    #     account — no email is disclosed;
    other = db.execute(
        select(AdminUser.id).where(
            AdminUser.telegram_user_id == telegram_id,
            AdminUser.id != admin_user.id,
        )
    ).scalar_one_or_none()
    if other is not None:
        raise AppError(
            409,
            "already_linked",
            "This Telegram account is already linked to a different console account.",
        )
    # (b) the console account is already linked to a different Telegram user
    #     (unlink on the console first; a leaked code must not silently
    #     transfer an account to a new owner).
    if admin_user.telegram_user_id is not None and admin_user.telegram_user_id != telegram_id:
        raise AppError(
            409,
            "already_linked",
            "This console account is already linked to a different Telegram account. "
            "Unlink it in the console first.",
        )

    admin_user.telegram_user_id = telegram_id
    link_code.revoked_at = datetime.now(timezone.utc)
    db.commit()
    return BotLinkResultOut(
        account_email=admin_user.email,
        account_name=admin_user.name,
        account_role=admin_user.role,
    )


def _looks_like_netscape_cookies(path) -> bool:
    """Netscape cookies.txt: a '# ...' header line, then tab-separated rows.

    A cookie line has 7 tab-separated fields (domain, flag, path, secure,
    expiry, name, value); some exports pad with a trailing empty field, so
    "at least 7" rather than exactly 7.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    lines = [ln.rstrip("\r\n") for ln in text.split("\n")]
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty or not non_empty[0].lstrip().startswith("#"):
        return False
    cookie_lines = [ln for ln in non_empty if not ln.lstrip().startswith("#")]
    return any(len(ln.split("\t")) >= 7 for ln in cookie_lines)


@router.post("/{telegram_id}/cookies", response_model=CookiesOut)
async def upload_cookies(
    telegram_id: int,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_bot_user),
) -> CookiesOut:
    limit_bytes = settings.max_cookies_kb * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit_bytes:
        raise AppError(
            413,
            "payload_too_large",
            f"Cookies file exceeds the {settings.max_cookies_kb} KB limit.",
            [{"limit_bytes": limit_bytes}],
        )

    dest = storage.cookie_file(telegram_id)
    tmp = dest.with_name(dest.name + ".tmp")
    written = 0
    try:
        with tmp.open("wb") as out:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit_bytes:
                    raise AppError(
                        413,
                        "payload_too_large",
                        f"Cookies file exceeds the {settings.max_cookies_kb} KB limit.",
                        [{"limit_bytes": limit_bytes}],
                    )
                out.write(chunk)
    except AppError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise AppError(500, "upload_failed", "Could not store the cookies file.") from exc

    if not _looks_like_netscape_cookies(tmp):
        tmp.unlink(missing_ok=True)
        raise AppError(
            422,
            "invalid_cookies_file",
            "That does not look like a Netscape cookies.txt export. Export one "
            "with a 'Get cookies.txt LOCALLY' style browser extension (or "
            "yt-dlp --cookies-from-browser) and send that file.",
            [{"field": "file"}],
        )

    os.replace(tmp, dest)
    # The file holds a live session: only its owner (and the service) may read it.
    os.chmod(dest, 0o600)
    digest = storage.sha256_file(dest)

    now = datetime.now(timezone.utc)
    row = _get_row(db, telegram_id)
    if row is None:
        # Consent is not a precondition here: the bot flow enforces the order
        # (agree, then upload), and a re-upload after a withdrawn consent must
        # not deadlock on a row the withdraw deleted.
        row = BotUser(telegram_user_id=telegram_id)
        db.add(row)
    row.cookies_sha256 = digest
    row.cookies_updated_at = now
    db.commit()
    db.refresh(row)
    # Log the path only; the contents are a live session and never logged.
    logger.info("cookies stored for telegram user %s at %s", telegram_id, dest)
    return CookiesOut(cookies_present=True, cookies_updated_at=row.cookies_updated_at)


@router.delete("/{telegram_id}", status_code=204)
def withdraw(
    telegram_id: int,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_bot_user),
) -> Response:
    storage.cookie_file(telegram_id).unlink(missing_ok=True)
    row = _get_row(db, telegram_id)
    if row is not None:
        db.delete(row)
        db.commit()
    return Response(status_code=204)


def require_session_consent(db: Session, principal: Principal, source_url: str) -> None:
    """Gate TikTok/Facebook ingestion on the user's own consented session.

    Called from `create_video` after the dedupe check: an already-known URL
    is returned to anyone, but a *new* gated download must run under the
    requesting user's cookies. The bot's service identity ("telegram:bot",
    no user id) and every non-telegram principal pass untouched.
    """
    if principal.role != "telegram":
        return
    prefix = "telegram:"
    if not principal.actor.startswith(prefix):
        return
    uid = principal.actor[len(prefix):]
    if not uid.isdigit():
        return  # "telegram:bot" is the service account, not a user session.
    if detect_platform(source_url) not in SESSION_GATED_PLATFORMS:
        return
    row = _get_row(db, int(uid))
    if row is not None and row.consented and storage.cookie_file(int(uid)).exists():
        return
    raise AppError(
        403,
        "consent_required",
        "Run /cookies in the bot to allow downloads with your own session.",
        [{"field": "source_url", "issue": "consent_required"}],
    )