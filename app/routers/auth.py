"""Auth routes."""
from __future__ import annotations

import httpx
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.auth import (
    Principal,
    create_access_token,
    get_principal,
    issue_guest_token,
    upsert_google_user,
    verify_password,
)
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import AdminUser, LinkCode
from app.routers.passcodes import CODE_ALPHABET, hash_code
from app.schemas import (
    GoogleCredentialRequest,
    GuestTokenOut,
    LoginRequest,
    LoginResponse,
    PrincipalOut,
    TelegramLinkOut,
    TelegramLinkStatusOut,
)
from app.security import enforce_rate_limit

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> LoginResponse:
    enforce_rate_limit(
        f"login:{_client_ip(request)}",
        settings.login_rate_limit,
        settings.login_rate_window,
    )
    if not payload.email or not payload.password:
        raise AppError(
            422,
            "validation_failed",
            "email and password are required.",
            [{"field": "email", "issue": "missing"}],
        )
    user = db.execute(
        select(AdminUser).where(AdminUser.email == payload.email)
    ).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise AppError(401, "unauthorized", "Invalid email or password.")
    # The role travels in the token for the client's convenience; the server
    # re-reads it from the database on every request (`auth._resolve_principal`),
    # so a changed role applies immediately, stale claim or not.
    token, expires = create_access_token(user.email, user.role)
    return LoginResponse(
        token=token,
        email=user.email,
        expires_at=expires,
        role=user.role,
        name=user.name,
    )


@router.post("/guest", response_model=GuestTokenOut)
def guest_session() -> GuestTokenOut:
    """An anonymous session for browsing public datasets (instruction.txt #3).

    Deliberately unauthenticated and scope-free: it can read public folders and
    nothing else. A private folder needs a passcode or share redemption, which
    returns a token scoped to that folder alone.
    """
    token, expires = issue_guest_token()
    return GuestTokenOut(token=token, role="guest", expires_at=expires)


@router.post("/google", response_model=LoginResponse)
def google_login(
    payload: GoogleCredentialRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> LoginResponse:
    """Sign in with a Google Identity Services ID token (instruction.txt #2).

    The browser gets the credential; the server verifies it with Google before
    trusting any claim. A Google account becomes a plain user, never an
    operator — promotions happen through the user-management surface.
    """
    if not settings.google_client_id:
        raise AppError(
            400,
            "google_not_configured",
            "Google sign-in is not configured on this server. Set GOOGLE_CLIENT_ID.",
        )
    enforce_rate_limit(
        f"login:{_client_ip(request)}",
        settings.login_rate_limit,
        settings.login_rate_window,
    )

    try:
        response = httpx.get(
            GOOGLE_TOKENINFO_URL,
            params={"id_token": payload.credential},
            timeout=8.0,
        )
    except httpx.HTTPError as exc:
        raise AppError(
            502, "upstream_error", "Could not reach Google to verify this sign-in."
        ) from exc

    if response.status_code != 200:
        raise AppError(401, "unauthorized", "Google sign-in could not be verified.")

    claims = response.json()
    if claims.get("aud") != settings.google_client_id:
        raise AppError(
            401, "unauthorized", "That Google credential was issued for another application."
        )
    if claims.get("email_verified") not in (True, "true"):
        raise AppError(401, "unauthorized", "This Google account's email is not verified.")
    sub = claims.get("sub")
    email = claims.get("email")
    if not sub or not email:
        raise AppError(401, "unauthorized", "Google did not return an account identity.")

    user = upsert_google_user(
        db,
        google_sub=str(sub),
        email=str(email),
        name=claims.get("name"),
        avatar_url=claims.get("picture"),
    )
    token, expires = create_access_token(user.email, user.role)
    return LoginResponse(
        token=token,
        email=user.email,
        expires_at=expires,
        role=user.role,
        name=user.name,
    )


@router.get("/me", response_model=PrincipalOut)
def me(
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> PrincipalOut:
    """Who the bearer token belongs to, for the console header."""
    user = None
    if principal.email:
        user = db.execute(
            select(AdminUser).where(AdminUser.email == principal.email)
        ).scalar_one_or_none()
    return PrincipalOut(
        actor=principal.actor,
        role=principal.role,
        email=principal.email,
        name=user.name if user else None,
        avatar_url=user.avatar_url if user else None,
        folder_ids=sorted(principal.scoped_folder_ids),
    )


# --------------------------------------------- Telegram account linking (bot)
# The console half of the link flow: sign in, generate a one-time code, hand
# it to the bot. The bot half (redeeming the code) lives in bot_users.py,
# authenticated by the service token instead of a session JWT.


def _require_console_account(db: Session, principal: Principal) -> AdminUser:
    """Only a signed-in console account (any account role) may manage its link.

    A guest token resolves fine through `get_principal`, and so does the bot's
    service identity — but neither names an AdminUser row, and a link code
    must always belong to a real account.
    """
    if not principal.email:
        raise AppError(
            403, "forbidden", "Sign in to a console account to manage its Telegram link."
        )
    user = db.execute(
        select(AdminUser).where(AdminUser.email == principal.email)
    ).scalar_one_or_none()
    if user is None:
        raise AppError(
            403, "forbidden", "Sign in to a console account to manage its Telegram link."
        )
    return user


@router.post("/telegram-link", response_model=TelegramLinkOut)
def create_telegram_link(
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> TelegramLinkOut:
    """Generate a one-time code that binds the caller's Telegram account.

    The code is 8 characters from the same readable alphabet as folder
    passcodes and only its sha256 hash is stored, so it is returned here
    exactly once and never again. Issuing a new code retires any previous
    outstanding one: at most one live code per account, so a code left on a
    screen cannot be redeemed after the user generates a fresh one.
    """
    user = _require_console_account(db, principal)
    now = datetime.now(timezone.utc)
    db.execute(
        update(LinkCode)
        .where(LinkCode.admin_user_id == user.id, LinkCode.revoked_at.is_(None))
        .values(revoked_at=now)
    )
    code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
    link = LinkCode(
        code_hash=hash_code(code),
        admin_user_id=user.id,
        expires_at=now + timedelta(minutes=settings.link_code_ttl_minutes),
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    deep_link = (
        f"https://t.me/{settings.telegram_bot_username}?start=link-{code}"
        if settings.telegram_bot_username
        else None
    )
    return TelegramLinkOut(code=code, deep_link=deep_link, expires_at=link.expires_at)


@router.get("/telegram-link", response_model=TelegramLinkStatusOut)
def get_telegram_link(
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> TelegramLinkStatusOut:
    """The caller's own Telegram link, for the console's account page."""
    user = _require_console_account(db, principal)
    return TelegramLinkStatusOut(telegram_user_id=user.telegram_user_id)


@router.delete("/telegram-link", status_code=204)
def delete_telegram_link(
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> None:
    """Withdraw the caller's own Telegram link (a no-op when unlinked)."""
    user = _require_console_account(db, principal)
    user.telegram_user_id = None
    db.commit()
