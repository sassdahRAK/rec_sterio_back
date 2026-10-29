"""Resolve actor strings into human-readable display names.

The raw actor string ("telegram:12345", "admin:a@b.c", "system") is the
authorization source of truth everywhere: `video_role` compares it by string
equality and the worker parses it back out of job payloads. Display is a
separate, read-only concern — resolved here, only where it is shown.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AdminUser


def display_name(db: Session, actor: Optional[str]) -> Optional[str]:
    """A readable name for an actor string, or None when it cannot be resolved.

    "telegram:<digits>" resolves through the account link so a video ingested
    by a bot user can show *who* they are on the console. Every other form
    ("admin:...", "member:...", "guest:...", "telegram:bot", "system", an
    unset value) has no single readable owner beyond what the raw string
    already says, so it resolves to None and the caller shows the raw string.
    """
    if not actor:
        return None
    prefix = "telegram:"
    if not actor.startswith(prefix):
        return None
    tail = actor[len(prefix):]
    if not tail.isdigit():
        return None  # "telegram:bot" is the service identity, not a user.
    user = db.execute(
        select(AdminUser).where(AdminUser.telegram_user_id == int(tail))
    ).scalar_one_or_none()
    if user is None:
        return None  # Not linked (or the link was withdrawn): nothing to show.
    return f"{user.name} ({user.email})" if user.name else user.email