"""Authentication: admin JWT bearer tokens + Telegram bot service token."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import AdminUser

logger = logging.getLogger("recsterio.auth")
ALGORITHM = "HS256"

# Header the Telegram bot uses to attribute a mutation to a user.
REQUESTED_BY_HEADER = "X-Requested-By"

bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """bcrypt hash. Uses the bcrypt library directly (passlib's bcrypt backend is
    incompatible with bcrypt >= 4.1 on Python 3.14)."""
    import bcrypt

    pw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    import bcrypt

    pw = password.encode("utf-8")[:72]
    try:
        return bcrypt.checkpw(pw, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(email: str) -> tuple[str, datetime]:
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_ttl_minutes
    )
    payload = {"sub": email, "iat": datetime.now(timezone.utc), "exp": expires}
    token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    return token, expires


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise AppError(401, "unauthorized", "Invalid or expired token.") from exc
    sub = payload.get("sub")
    if not sub:
        raise AppError(401, "unauthorized", "Invalid or expired token.")
    return str(sub)


def seed_admin_user(db: Session) -> Optional[AdminUser]:
    """Create the bootstrap admin from ADMIN_EMAIL/ADMIN_PASSWORD if not present."""
    email = (settings.admin_email or "").strip().lower()
    password = settings.admin_password
    if not email or not password:
        logger.warning("ADMIN_EMAIL/ADMIN_PASSWORD not set; skipping admin bootstrap.")
        return None
    existing = db.execute(
        select(AdminUser).where(AdminUser.email == email)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    user = AdminUser(
        email=email, password_hash=hash_password(password), is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Seeded admin user %s", email)
    return user


def resolve_actor(
    request: Request, token: str, credentials: Optional[HTTPAuthorizationCredentials], db: Session
) -> str:
    """Return the actor string: 'admin:<email>' or 'telegram:<id>' or 'system'."""
    requested_by = request.headers.get(REQUESTED_BY_HEADER)
    if requested_by:
        return requested_by.strip()[:128]

    # Service token for the Telegram bot.
    if settings.bot_api_token and token == settings.bot_api_token:
        return "telegram:bot"

    email = decode_access_token(token)
    user = db.execute(
        select(AdminUser).where(AdminUser.email == email)
    ).scalar_one_or_none()
    if user is None or not user.is_active:
        raise AppError(401, "unauthorized", "Invalid or expired token.")
    return f"admin:{user.email}"


def get_actor(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> str:
    if credentials is None or not credentials.credentials:
        raise AppError(
            401, "unauthorized", "Missing bearer token.", [{"header": "Authorization"}]
        )
    return resolve_actor(request, credentials.credentials, credentials, db)


def require_admin(
    actor: str = Depends(get_actor),
) -> str:
    """Some routes are admin-only (not used by the bot)."""
    if not actor.startswith("admin:"):
        raise AppError(403, "forbidden", "This action requires an admin token.")
    return actor
