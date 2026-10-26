"""Authentication: admin JWT bearer tokens + Telegram bot service token."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional

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


def create_access_token(
    email: str,
    role: str = "admin",
    folder_ids: Optional[list] = None,
    ttl_minutes: Optional[int] = None,
    folder_roles: Optional[dict] = None,
) -> tuple[str, datetime]:
    """Issue a bearer token.

    `role` is 'admin' | 'member' | 'guest'. A guest token carries `folders` —
    the only folders it may read — and the role it was granted on each, so a
    passcode or share link grants exactly the access it was created for and
    nothing else.
    """
    minutes = settings.access_token_ttl_minutes if ttl_minutes is None else ttl_minutes
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=minutes)
    payload = {
        "sub": email,
        "role": role,
        "folders": [str(f) for f in (folder_ids or [])],
        "roles": {str(k): str(v) for k, v in (folder_roles or {}).items()},
        "iat": now,
        "exp": expires,
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    return token, expires


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise AppError(401, "unauthorized", "Invalid or expired token.") from exc
    if not payload.get("sub"):
        raise AppError(401, "unauthorized", "Invalid or expired token.")
    return payload


def decode_access_token(token: str) -> str:
    """Backwards-compatible helper returning just the subject."""
    return str(decode_token(token)["sub"])


def issue_guest_token() -> tuple[str, datetime]:
    """A token for a visitor who has not signed in (instruction.txt #3)."""
    return create_access_token(
        f"guest:{uuid.uuid4().hex[:12]}", role="guest", ttl_minutes=settings.guest_ttl_minutes
    )


def issue_folder_token(
    folder_id: uuid.UUID, role: str = "viewer"
) -> tuple[str, datetime]:
    """A token scoped to exactly one folder (passcode or share-link redemption)."""
    return create_access_token(
        f"guest:{uuid.uuid4().hex[:12]}",
        role="guest",
        folder_ids=[folder_id],
        folder_roles={folder_id: role},
        ttl_minutes=settings.guest_ttl_minutes,
    )


def upsert_google_user(
    db: Session, google_sub: str, email: str, name: Optional[str] = None,
    avatar_url: Optional[str] = None,
) -> AdminUser:
    """Create or refresh the local user for a verified Google identity."""
    email = (email or "").strip().lower()
    user = db.execute(
        select(AdminUser).where(AdminUser.google_sub == google_sub)
    ).scalar_one_or_none()
    if user is None and email:
        user = db.execute(select(AdminUser).where(AdminUser.email == email)).scalar_one_or_none()
    if user is None:
        # A Google account is a dataset *member*, never an operator. The seeded
        # bootstrap account stays the only 'admin'.
        user = AdminUser(
            email=email or f"{google_sub}@google.user",
            google_sub=google_sub,
            role="member",
        )
        db.add(user)
    elif not user.role:
        user.role = "member"
    user.google_sub = google_sub
    if email:
        user.email = email
    if name:
        user.name = name
    if avatar_url:
        user.avatar_url = avatar_url
    user.is_active = True
    db.commit()
    db.refresh(user)
    return user


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


@dataclass(frozen=True)
class Principal:
    """Who is calling, and which folders a scoped token lets them see."""

    actor: str
    role: str
    email: Optional[str] = None
    scoped_folder_ids: frozenset = frozenset()
    # folder id -> role granted by the passcode / share link that issued the token
    scoped_roles: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_signed_in(self) -> bool:
        return self.role in ("admin", "member")

    @property
    def is_guest(self) -> bool:
        return self.role == "guest"

    def can_access(self, folder_id) -> bool:
        return str(folder_id) in self.scoped_folder_ids

    def role_on(self, folder_id) -> Optional[str]:
        return self.scoped_roles.get(str(folder_id))


def actor_role(actor: str) -> str:
    """'admin:a@b.c' -> 'admin'; 'guest:abc' -> 'guest'; 'telegram:bot' -> 'telegram'."""
    return actor.split(":", 1)[0] if ":" in actor else actor


def _resolve_principal(
    request: Request, token: str, db: Session, required: bool = True
) -> Principal:
    requested_by = request.headers.get(REQUESTED_BY_HEADER)
    if requested_by:
        actor = requested_by.strip()[:128]
        return Principal(actor, actor_role(actor), actor.split(":", 1)[1] or None)

    # Service token for the Telegram bot.
    if settings.bot_api_token and token == settings.bot_api_token:
        return Principal("telegram:bot", "telegram")

    if not token:
        if required:
            raise AppError(
                401, "unauthorized", "Missing bearer token.", [{"header": "Authorization"}]
            )
        return Principal("guest:anonymous", "guest")

    payload = decode_token(token)
    role = payload.get("role") or "admin"
    sub = str(payload.get("sub"))

    # A guest token is self-describing; it names no account row.
    if role == "guest":
        return Principal(
            f"guest:{sub}",
            "guest",
            None,
            frozenset(payload.get("folders") or []),
            dict(payload.get("roles") or {}),
        )

    user = db.execute(
        select(AdminUser).where(AdminUser.email == sub)
    ).scalar_one_or_none()
    if user is None or not user.is_active:
        raise AppError(401, "unauthorized", "Invalid or expired token.")
    return Principal(f"{role}:{user.email}", role, user.email)


def resolve_actor(
    request: Request,
    token: str,
    credentials: Optional[HTTPAuthorizationCredentials],
    db: Session,
    required: bool = True,
) -> str:
    """Return the actor string: 'admin:<email>' | 'member:<email>' | 'guest:<id>' | 'telegram:bot'."""
    return _resolve_principal(request, token, db, required).actor


def get_actor(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> str:
    token = credentials.credentials if credentials else ""
    return resolve_actor(request, token, credentials, db)


def get_optional_actor(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> str:
    """Like `get_actor`, but an absent token resolves to an anonymous guest.

    Used by the public dataset surface (instruction.txt #3): a visitor who has
    not signed in may still read public folders.
    """
    token = credentials.credentials if credentials else ""
    return resolve_actor(request, token, credentials, db, required=False)


def get_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Principal:
    return _resolve_principal(
        request, credentials.credentials if credentials else "", db, required=True
    )


def get_optional_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Principal:
    """Principal for routes a guest may also reach (public dataset browsing)."""
    return _resolve_principal(
        request, credentials.credentials if credentials else "", db, required=False
    )


def require_admin(
    actor: str = Depends(get_actor),
) -> str:
    """Some routes are admin-only (not used by the bot)."""
    if not actor.startswith("admin:"):
        raise AppError(403, "forbidden", "This action requires an admin token.")
    return actor
