"""Passcodes: the door code a private folder can be opened with (instruction.txt #4, #7).

The owner sets the code, how long it lives, whether it burns after one use and
how many devices may hold it. Codes are stored hashed — the API never returns
one — so a database leak does not leak the doors.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.access import folder_or_404, require_manage
from app.auth import Principal, get_principal
from app.db import get_db
from app.errors import AppError
from app.models import Folder, FolderPasscode, PasscodeGrant
from app.schemas import PasscodeCreate, PasscodeOut

router = APIRouter(tags=["passcodes"])


# Ambiguous characters (0/O, 1/I/L) are omitted so a code can be read aloud.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def generate_code() -> str:
    """A readable 8-character code, grouped for dictation: 'ABCD-2345'."""
    raw = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


def hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


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
        code_hash=hash_code(code),
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
    db: Session, folder_id: UUID, code: str, device_key: str | None
) -> Folder:
    """Validate a passcode and consume one use. Returns the opened folder.

    Raises AppError(401) for every failure so a wrong code, an expired code and
    a revoked code are indistinguishable to the caller.
    """
    now = datetime.now(timezone.utc)
    wanted = hash_code(code)

    candidates = db.execute(
        select(FolderPasscode).where(
            FolderPasscode.folder_id == folder_id,
            FolderPasscode.revoked_at.is_(None),
        )
    ).scalars().all()
    passcode = next((p for p in candidates if p.code_hash == wanted), None)

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
                    folder_id=folder_id,
                    device_key_hash=device_hash,
                )
            )

    passcode.use_count = (passcode.use_count or 0) + 1
    if passcode.one_time:
        # Burn it: a one-time code must not open the door twice, even if the
        # same device comes back.
        passcode.revoked_at = now
    db.commit()

    return folder_or_404(db, folder_id)
