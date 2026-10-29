"""Passcodes: the door code a private folder can be opened with (instruction.txt #4, #7).

The owner sets the code, how long it lives, whether it burns after one use and
how many devices may hold it. Codes are stored hashed — the API never returns
one — so a database leak does not leak the doors.

Storage is bcrypt (per-hash salt, slow to brute-force). Rows written before
the switch hold unsalted sha256 instead; `verify_code` accepts both, and a
legacy row is lazily rehash-stored when its code actually opens a door, so
old codes convert themselves on first use and nothing needs a data migration.
`hash_code` (plain sha256) stays for the Telegram link codes, which are
looked up by exact hash and are not brute-faced from the outside.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from hmac import compare_digest
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.access import folder_or_404, require_manage
from app.auth import Principal, get_principal, hash_password
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import Folder, FolderPasscode, PasscodeGrant
from app.schemas import PasscodeCreate, PasscodeOut

router = APIRouter(tags=["passcodes"])


# Ambiguous characters (0/O, 1/I/L) are omitted so a code can be read aloud.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

# Default for the most recent live candidate rows a single redemption verifies.
# bcrypt is deliberately slow, so the verify loop is paid once per candidate:
# without a cap, a folder (or an instance) stacked with live passcodes turned
# one wrong guess into N full bcrypt runs — CPU amplification anyone
# unauthenticated can trigger. Newest-first keeps every code written in the
# current era redeemable; the residual surface is documented on
# `redeem_passcode`. Operators tune the live value with
# `PUBLIC_REDEEM_SCAN_LIMIT` (`settings.public_redeem_scan_limit`).
CANDIDATE_LIMIT = 100


def generate_code() -> str:
    """A readable 8-character code, grouped for dictation: 'ABCD-2345'."""
    raw = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


def hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def new_code_hash(code: str) -> str:
    """How a passcode is stored going forward: bcrypt (salted, cost-bounded)."""
    return hash_password(code.strip())


def _looks_sha256(value: str) -> bool:
    """True for a legacy unsalted-sha256 passcode hash (64 hex chars)."""
    return (
        len(value) == 64
        and value == value.lower()
        and all(c in "0123456789abcdef" for c in value)
    )


def verify_code(code: str, stored_hash: str) -> bool:
    """Dual-verify a presented code against a stored passcode hash.

    bcrypt first (all new rows); a sha256 row falls back to the legacy digest.
    The caller decides what to do with the storage scheme afterwards —
    `redeem_passcode` rehash-stores a legacy row that passed.
    """
    code = code.strip()
    if stored_hash.startswith("$2"):
        import bcrypt

        try:
            return bcrypt.checkpw(code.encode("utf-8")[:72], stored_hash.encode("utf-8"))
        except (ValueError, TypeError):
            return False
    if _looks_sha256(stored_hash):
        return compare_digest(hash_code(code), stored_hash)
    return False


def hash_device_key(device_key: str) -> str:
    return hashlib.sha256(device_key.strip().encode("utf-8")).hexdigest()


def _get_passcode(db: Session, passcode_id: UUID) -> FolderPasscode:
    passcode = db.get(FolderPasscode, passcode_id)
    if passcode is None:
        raise AppError(
            404, "not_found", "Passcode not found.", [{"id": str(passcode_id)}]
        )
    return passcode


def _is_expired(passcode: FolderPasscode, now: datetime) -> bool:
    return passcode.expires_at is not None and passcode.expires_at <= now


def _is_exhausted(passcode: FolderPasscode) -> bool:
    return passcode.max_uses is not None and passcode.use_count >= passcode.max_uses


def _device_grants(db: Session, passcode_id: UUID) -> list[PasscodeGrant]:
    return list(
        db.execute(
            select(PasscodeGrant).where(PasscodeGrant.passcode_id == passcode_id)
        ).scalars().all()
    )


@router.post("/folders/{folder_id}/passcodes", response_model=PasscodeOut, status_code=201)
def create_passcode(
    folder_id: UUID,
    payload: PasscodeCreate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> PasscodeOut:
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)

    # Omit `code` to have one generated; it is returned exactly once, here.
    generated = payload.code is None
    code = payload.code or generate_code()

    passcode = FolderPasscode(
        folder_id=folder.id,
        code_hash=new_code_hash(code),
        label=payload.label,
        expires_at=payload.expires_at,
        max_uses=payload.max_uses,
        max_devices=payload.max_devices,
        one_time=payload.one_time,
        created_by=principal.actor,
    )
    db.add(passcode)
    db.commit()
    db.refresh(passcode)
    out = PasscodeOut.model_validate(passcode)
    if generated:
        out.code = code
    return out


@router.get("/folders/{folder_id}/passcodes", response_model=list[PasscodeOut])
def list_passcodes(
    folder_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[PasscodeOut]:
    folder = folder_or_404(db, folder_id)
    require_manage(db, folder, principal)
    items = db.execute(
        select(FolderPasscode)
        .where(FolderPasscode.folder_id == folder_id)
        .order_by(FolderPasscode.created_at.desc())
    ).scalars().all()
    out: list[PasscodeOut] = []
    for passcode in items:
        row = PasscodeOut.model_validate(passcode)
        row.device_count = len(_device_grants(db, passcode.id))
        out.append(row)
    return out


@router.delete("/passcodes/{passcode_id}", status_code=204)
def revoke_passcode(
    passcode_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> None:
    passcode = _get_passcode(db, passcode_id)
    require_manage(db, folder_or_404(db, passcode.folder_id), principal)
    passcode.revoked_at = datetime.now(timezone.utc)
    db.commit()


def redeem_passcode(
    db: Session,
    code: str,
    device_key: str | None,
    folder_id: UUID | None = None,
    folder_name: str | None = None,
) -> Folder:
    """Validate a passcode and consume one use. Returns the opened folder.

    The folder is identified by the id from a link, by its name typed in, or by
    the code alone — never by listing folders. A public list of folders would
    tell every visitor how many folders exist and what they are called, which
    is exactly what a private folder must not disclose.

    Raises AppError(401) for every failure so a wrong code, an expired code and
    a revoked code are indistinguishable to the caller.

    Cost model and residual brute-force surface: a wrong guess pays one bcrypt
    verify per live candidate row, and the candidate set is capped — at the
    `public_redeem_scan_limit` newest live rows matching the named target (`folder_id`
    when present, else the whole instance for a code-only search). Codes older
    than that window no longer redeem here and must be revoked/reissued; in
    exchange the per-request verify cost (and with it the unauthenticated CPU
    amplification) is bounded no matter how many rows pile up. Guessing is
    additionally throttled per client IP (`/public/*` redemption routes).
    """
    now = datetime.now(timezone.utc)

    # bcrypt hashes are salted, so the code cannot be looked up by an SQL
    # equality: verify the candidate rows in Python instead, newest-first and
    # capped (see `settings.public_redeem_scan_limit`) — a wrong guess pays
    # only the capped verify cost.
    statement = select(FolderPasscode).where(FolderPasscode.revoked_at.is_(None))
    if folder_id is not None:
        statement = statement.where(FolderPasscode.folder_id == folder_id)
    if folder_name:
        statement = statement.join(
            Folder, Folder.id == FolderPasscode.folder_id
        ).where(func.lower(Folder.name) == folder_name.strip().lower())
    statement = statement.order_by(FolderPasscode.created_at.desc()).limit(
        settings.public_redeem_scan_limit
    )

    matches = [
        passcode
        for passcode in db.execute(statement).scalars().all()
        if verify_code(code, passcode.code_hash)
    ]
    if len(matches) > 1:
        # Two folders share this code and the caller named neither precisely.
        # Say only that, never which folders matched.
        raise AppError(
            422,
            "ambiguous_folder",
            "More than one folder matches. Open the link your host sent you instead.",
        )
    passcode = matches[0] if matches else None

    if passcode is None:
        raise AppError(401, "unauthorized", "That passcode is not valid.")
    if _is_expired(passcode, now):
        raise AppError(401, "unauthorized", "That passcode has expired.")
    if _is_exhausted(passcode):
        raise AppError(401, "unauthorized", "That passcode has been used up.")

    if passcode.max_devices is not None:
        if not device_key:
            # A device cap cannot be enforced without a device identity.
            raise AppError(
                422,
                "validation_failed",
                "This passcode requires a device identifier.",
                [{"field": "device_key", "issue": "required"}],
            )
        device_hash = hash_device_key(device_key)
        grants = _device_grants(db, passcode.id)
        known = any(g.device_key_hash == device_hash for g in grants)
        if not known and len(grants) >= passcode.max_devices:
            raise AppError(
                401,
                "unauthorized",
                "This passcode has reached its device limit.",
                [{"limit": passcode.max_devices}],
            )
        if not known:
            db.add(
                PasscodeGrant(
                    passcode_id=passcode.id,
                    folder_id=passcode.folder_id,
                    device_key_hash=device_hash,
                )
            )

    # Lazy legacy migration: this row opened a door, so its unsalted sha256
    # hash is replaced here with bcrypt. Codes that never verify stay put,
    # meaning no data migration is needed and nothing is stored worse.
    if _looks_sha256(passcode.code_hash):
        passcode.code_hash = new_code_hash(code)

    passcode.use_count = (passcode.use_count or 0) + 1
    if passcode.one_time:
        # Burn it: a one-time code must not open the door twice, even if the
        # same device comes back.
        passcode.revoked_at = now
    db.commit()

    return folder_or_404(db, passcode.folder_id)
