"""Auth routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import create_access_token, verify_password
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import AdminUser
from app.schemas import LoginRequest, LoginResponse
from app.security import enforce_rate_limit

router = APIRouter(prefix="/auth", tags=["auth"])


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
    token, expires = create_access_token(user.email)
    return LoginResponse(token=token, email=user.email, expires_at=expires)
