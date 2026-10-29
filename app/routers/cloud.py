"""Personal-cloud storage: a contributor's own Google Drive / OneDrive.

Three surfaces:

* **Connect / status / disconnect** — the contributor links a drive from the
  portal's Settings. We exchange the provider's authorization code for a
  renewable refresh token and store exactly one link per (account, provider).
* **Cloud upload** — on ingest a signed-in contributor may choose to keep the
  bytes on their own cloud. The API returns a resumable upload session; the
  client uploads the bytes directly to the provider, then calls `complete`,
  and the normal pipeline (audio extract → transcribe) runs from a temp copy.
* **Streaming** — `videos.py`/`zip_export.py` read cloud bytes through
  `cloud_providers.open_download` exactly like local files.

Revocation is deliberate: disconnecting (or a dead grant) leaves the video row
and its transcript intact but marks the unit "storage unavailable", never a 500.
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import schemas
from app.schemas import CompleteCloudUploadIn
from app.auth import Principal, get_principal
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import AdminUser, DriveLink, Video
from app.services import asr, cloud_providers, storage
from app.services.cloud_providers import decrypt_token, encrypt_token
from app.worker import queue

logger = logging.getLogger("recsterio.cloud")

router = APIRouter(prefix="/cloud", tags=["cloud"])

# Ingested files land in a dedicated app folder on the user's drive; this name
# keeps them together and identifiable without touching the user's own files.
REMOTE_APP_FOLDER = "RecSterio"


def _verify_drive_file(provider: str, access_token: str, file_id: str) -> None:
    """Verify the file exists in the caller's drive and is accessible."""
    from app.services.http_client import http_json
    if provider == "google":
        status, _, data = http_json(
            "GET",
            f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=id,name,size",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if status == 404:
            raise AppError(404, "cloud_file_not_found", "The specified file was not found in your Google Drive.")
        if status == 403:
            raise AppError(403, "cloud_file_forbidden", "You do not have access to this file in Google Drive.")
        if status >= 400:
            raise AppError(422, "cloud_file_invalid", "Could not verify the file in your Google Drive.")
    else:
        status, _, data = http_json(
            "GET",
            f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if status == 404:
            raise AppError(404, "cloud_file_not_found", "The specified file was not found in your OneDrive.")
        if status == 403:
            raise AppError(403, "cloud_file_forbidden", "You do not have access to this file in OneDrive.")
        if status >= 400:
            raise AppError(422, "cloud_file_invalid", "Could not verify the file in your OneDrive.")


def _user_for(principal: Principal, db: Session) -> AdminUser:
    """The cloud surface is for signed-in accounts only — a guest has no drive
    row to attach a link to, and the upload path is a contributor feature."""
    if not principal.email:
        raise AppError(401, "unauthorized", "Sign in to use your own cloud storage.")
    user = db.execute(
        select(AdminUser).where(AdminUser.email == principal.email)
    ).scalar_one_or_none()
    if user is None:
        raise AppError(401, "unauthorized", "Sign in to use your own cloud storage.")
    return user


def _link_for(user_id, provider: str, db: Session) -> Optional[DriveLink]:
    return db.execute(
        select(DriveLink).where(DriveLink.user_id == user_id, DriveLink.provider == provider)
    ).scalar_one_or_none()


def _checked_redirect(supplied: Optional[str]) -> str:
    """The redirect the token exchange must echo back.

    The provider rejects a mismatch with the consent screen, so the client's
    own value is honored — but only on the configured portal origin, or a
    caller could talk the server into exchanging a code issued for a foreign
    redirect. The configured default is used when nothing is supplied.
    """
    configured = settings.drive_oauth_redirect or ""
    candidate = (supplied or "").strip() or configured
    if not candidate:
        raise AppError(400, "provider_not_configured", "No OAuth redirect is configured.")
    try:
        from urllib.parse import urlsplit

        want = urlsplit(configured)
        got = urlsplit(candidate)
        if configured and (want.scheme, want.netloc) != (got.scheme, got.netloc):
            raise AppError(
                422, "validation_failed",
                "redirect_uri must be on the portal's own origin.",
                [{"field": "redirect_uri", "issue": "foreign_origin"}],
            )
    except AppError:
        raise
    return candidate


def _require_link(user: AdminUser, provider: str, db: Session) -> DriveLink:
    link = _link_for(user.id, provider, db)
    if link is None:
        raise AppError(
            409, "drive_not_connected",
            f"Connect your {'Google Drive' if provider == 'google' else 'OneDrive'} first.",
            [{"provider": provider}],
        )
    return link


# ------------------------------------------------------------------ connect


@router.get("/providers", response_model=schemas.DriveProviderStatus)
def provider_status() -> schemas.DriveProviderStatus:
    """Which providers this deployment offers (both OAuth apps configured)."""
    available = cloud_providers.providers_available()
    return schemas.DriveProviderStatus(**available)


@router.get("/links", response_model=list[schemas.DriveLinkOut])
def list_links(
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[schemas.DriveLinkOut]:
    user = _user_for(principal, db)
    rows = db.execute(
        select(DriveLink).where(DriveLink.user_id == user.id).order_by(DriveLink.created_at.asc())
    ).scalars().all()
    return [
        schemas.DriveLinkOut(
            id=row.id,
            provider=row.provider,
            account_email=row.account_email,
            connected_at=row.created_at,
        )
        for row in rows
    ]


@router.post("/links", response_model=schemas.DriveLinkOut, status_code=201)
def connect_drive(
    payload: schemas.DriveConnectIn,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> schemas.DriveLinkOut:
    """Exchange a provider authorization code for a renewable grant."""
    user = _user_for(principal, db)
    redirect_uri = _checked_redirect(payload.redirect_uri)
    grant = cloud_providers.exchange_code(payload.provider, payload.code, redirect_uri)
    # One live link per (user, provider): reconnecting replaces the tokens.
    link = _link_for(user.id, payload.provider, db)
    if link is None:
        link = DriveLink(user_id=user.id, provider=payload.provider, account_email="", refresh_token="")
        db.add(link)
    link.account_email = grant["account_email"]
    link.account_subject = grant.get("account_subject")
    link.refresh_token = encrypt_token(grant["refresh_token"])
    db.commit()
    db.refresh(link)
    return schemas.DriveLinkOut(
        id=link.id, provider=link.provider, account_email=link.account_email, connected_at=link.created_at
    )


@router.delete("/links/{provider}", status_code=204)
def disconnect_drive(
    provider: str,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> None:
    """Detach a drive. Videos already stored there keep their metadata and
    transcript but become 'storage unavailable' until reconnected."""
    user = _user_for(principal, db)
    link = _link_for(user.id, provider, db)
    if link is None:
        raise AppError(404, "not_found", "That drive is not connected.")
    db.delete(link)
    db.commit()


# --------------------------------------------------------------- cloud upload


def _remote_name(filename: Optional[str], video_id: UUID) -> str:
    base = Path(filename or "upload").name or "upload"
    stem, dot, suffix = base.rpartition(".")
    safe_suffix = f".{suffix}" if dot and len(suffix) <= 8 else ""
    # A content hash prefix keeps names unique without exposing the uuid twice.
    return f"{REMOTE_APP_FOLDER}/{video_id.hex[:12]}-{hashlib.sha1(str(video_id).encode()).hexdigest()[:8]}{safe_suffix}"


@router.post("/uploads", response_model=schemas.CloudUploadOut, status_code=201)
def start_cloud_upload(
    payload: schemas.CloudUploadIn,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> schemas.CloudUploadOut:
    """Open a resumable upload session on the contributor's own drive.

    Creates the Video row (status `uploading`, no bytes on the server yet) and
    returns where the client PUTs the bytes. `complete` is called afterwards.
    """
    user = _user_for(principal, db)
    link = _require_link(user, payload.backend, db)
    if not cloud_providers.providers_available().get(payload.backend):
        raise AppError(400, "provider_not_configured", "That provider is not available here.")

    mime = mimetypes.guess_type(payload.filename or "")[0] or "application/octet-stream"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    declared_size = min(payload.size, max_bytes) if payload.size else max_bytes

    video = Video(
        source_url=f"upload://{(payload.filename or 'upload').split('/')[-1]}",
        platform="direct",
        title=None,
        status="uploading",
        progress=0,
        created_by=principal.actor,
        storage_backend=payload.backend,
        drive_link_id=link.id,
        mime_type=mime,
    )
    db.add(video)
    db.flush()

    access = cloud_providers.refresh_access_token(payload.backend, decrypt_token(link.refresh_token))
    remote = _remote_name(payload.filename, video.id)
    target = cloud_providers.create_upload_target(payload.backend, access, remote, mime, declared_size)
    # The provider file id is known only after the bytes land; `complete` sets it.
    return schemas.CloudUploadOut(
        video_id=video.id,
        upload_url=target.upload_url,
        upload_headers=target.upload_headers,
        expires_at=target.expires_at,
    )


@router.post("/uploads/{video_id}/complete", response_model=schemas.VideoOut)
def complete_cloud_upload(
    video_id: UUID,
    payload: "CompleteCloudUploadIn",
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> schemas.VideoOut:
    """The bytes are on the contributor's drive: finish the row and run the pipeline.

    Downloads the file to a temp copy to extract audio + transcribe, then removes
    the temp copy (the authoritative bytes stay in the user's drive). Sets the
    video `downloaded` and enqueues the usual audio/transcribe chain.
    """
    video = db.get(Video, video_id)
    if video is None:
        raise AppError(404, "not_found", "Video not found.", [{"id": str(video_id)}])
    if video.created_by != principal.actor and principal.role != "super_admin":
        raise AppError(403, "forbidden", "You can only complete your own upload.")
    if video.storage_backend is None:
        raise AppError(409, "conflict", "This video is not a cloud upload.", [{"id": str(video_id)}])
    if video.drive_file_id:
        raise AppError(409, "conflict", "This upload was already completed.", [{"id": str(video_id)}])

    link = db.get(DriveLink, video.drive_link_id) if video.drive_link_id else None
    if link is None:
        raise AppError(409, "drive_not_connected", "Reconnect the drive to finish this upload.")

    # Materialize a temp copy so the existing extract/transcribe pipeline runs
    # unchanged; the copy is removed by `_ensure_audio` once audio.wav exists,
    # and the contributor's drive stays the authoritative source of the bytes.
    temp_media = storage.cloud_media_file(video.id)
    try:
        access = cloud_providers.refresh_access_token(video.storage_backend, decrypt_token(link.refresh_token))
        # Verify the file exists in the caller's own drive before trusting the
        # supplied provider_file_id.  This prevents one user from pointing a
        # video row at a file they do not own.
        _verify_drive_file(video.storage_backend, access, payload.provider_file_id)

        video.drive_file_id = payload.provider_file_id
        video.file_size = payload.size
        video.checksum_sha256 = payload.checksum_sha256
        video.status = "downloading"
        db.flush()

        read = cloud_providers.open_download(video.storage_backend, access, video.drive_file_id)
        with temp_media.open("wb") as out:
            for chunk in read.body:
                out.write(chunk)
        if not video.mime_type:
            video.mime_type = (
                read.headers.get("Content-Type")
                or mimetypes.guess_type(temp_media.name)[0]
                or "application/octet-stream"
            )
        video.local_path = str(temp_media)
        video.duration_seconds = asr.probe_duration(temp_media)
        video.status = "downloaded"
        video.progress = 100
        if video.file_size is None:
            video.file_size = temp_media.stat().st_size
        db.flush()
        queue.enqueue(
            db,
            "extract_audio",
            video_id=video.id,
            payload={"chain": ["transcribe"]},
            priority=100,
        )
        db.commit()
        db.refresh(video)
        return schemas.VideoOut.model_validate(video)
    except AppError:
        db.rollback()
        temp_media.unlink(missing_ok=True)
        raise
    except Exception:
        db.rollback()
        logger.exception("cloud upload completion failed for %s", video_id)
        temp_media.unlink(missing_ok=True)
        raise AppError(500, "cloud_upload_failed", "Could not process the uploaded file from your drive.")
