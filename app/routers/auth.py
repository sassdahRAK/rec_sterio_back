"""Auth routes."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
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
from app.models import AdminUser
from app.schemas import (
    GoogleCredentialRequest,
    GuestTokenOut,
    LoginRequest,
    LoginResponse,
    PrincipalOut,
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
    # The role must go into the token, not just the response body: every
    # authorization decision reads the JWT, so a member signed in with an
    # 'admin' token would be an admin.
    token, expires = create_access_token(user.email, user.role or "admin")
    return LoginResponse(
        token=token,
        email=user.email,
        expires_at=expires,
        role=user.role or "admin",
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
    trusting any claim. A Google account becomes a *member*, never an operator —
    the seeded bootstrap account is the only admin.
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
    token, expires = create_access_token(user.email, user.role or "member")
    return LoginResponse(
        token=token,
        email=user.email,
        expires_at=expires,
        role=user.role or "member",
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
