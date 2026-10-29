"""Personal-cloud providers: Google Drive and OneDrive.

A contributor may keep their video bytes on their own cloud instead of the
server disk. This module is the only place that talks to the two providers —
it exchanges the OAuth authorization code for tokens, opens upload sessions in
a dedicated "RecSterio" folder, and builds ranged stream readers for a stored
file so the media/audio/ZIP routes can serve cloud bytes like local ones.

Provider calls are wrapped in `http_json`/`http_bytes` (the codebase's shared
helpers), which raise the codebase's `errors.upstream`, keeping every provider
failure one place and the routers thin. Base URLs are module constants so the
test suite can point them at a fake provider. No new dependency: plain urllib.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from app.config import settings
from app.errors import AppError, upstream
from app.services.http_client import http_bytes, http_json

logger = logging.getLogger("recsterio.cloud")

__all__ = [
    "PROVIDER_NAMES",
    "CloudAuthError",
    "UploadTarget",
    "CloudRead",
    "providers_available",
    "exchange_code",
    "refresh_access_token",
    "create_upload_target",
    "open_download",
]

# Overridable in tests so the suite never reaches the real providers.
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_UPLOAD_SESSION_URL = "https://www.googleapis.com/upload/drive/v3/files"
GOOGLE_DOWNLOAD_URL = "https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
MICROSOFT_OAUTH_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
MICROSOFT_GRAPH_ME_URL = "https://graph.microsoft.com/v1.0/me"
MICROSOFT_GRAPH_UPLOAD_URL = "https://graph.microsoft.com/v1.0/me/drive/special/approot:/{path}:/createUploadSession"
MICROSOFT_GRAPH_DOWNLOAD_URL = "https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content"

DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
ONEDRIVE_SCOPE = "offline_access https://graph.microsoft.com/Files.ReadWrite"

PROVIDER_NAMES = ("google", "onedrive")


class CloudAuthError(AppError):
    """The provider rejected the consent code or the grant is unusable."""

    def __init__(self, message: str):
        super().__init__(400, "cloud_auth_rejected", message)


# --------------------------------------------------------------------- oauth


def _form_post(url: str, fields: Dict[str, str], provider: str) -> Dict[str, Any]:
    status, _, payload = http_json(
        "POST",
        url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=fields,
    )
    if status >= 400:
        detail = payload.get("error_description") or payload.get("error") or str(status)
        # A bad/expired code is the caller's problem to surface; a 5xx is ours.
        if status in (400, 401):
            raise CloudAuthError(f"{provider} rejected the authorization: {detail}.")
        raise upstream(provider, f"token endpoint answered {status}", str(detail))
    return payload


def _require_grant(token: Dict[str, Any], provider: str) -> str:
    access = token.get("access_token")
    if not access:
        raise CloudAuthError(f"{provider} returned no access token.")
    return access


def _require_refresh(token: Dict[str, Any], provider: str) -> str:
    refresh = token.get("refresh_token")
    if not refresh:
        raise CloudAuthError(
            f"{provider} returned no refresh token — request the consent with "
            "'access_type=offline' / 'prompt=consent' so the grant is renewable."
        )
    return refresh


def exchange_code(provider: str, code: str, redirect_uri: str) -> Dict[str, Any]:
    """Trade the provider's authorization code for a renewable grant.

    Returns {refresh_token, account_email, account_subject?}. The caller stores
    the refresh token; the access token is used once here (to read the account
    identity) and then discarded.
    """
    if provider == "google":
        if not settings.google_client_id or not settings.google_client_secret:
            raise AppError(400, "provider_not_configured", "Google Drive storage is not configured on this server.")
        token = _form_post(
            GOOGLE_OAUTH_TOKEN_URL,
            {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
            "google",
        )
        access = _require_grant(token, "Google")
        status, _, info = http_json("GET", GOOGLE_USERINFO_URL, headers=_bearer(access))
        if status >= 400:
            raise upstream("google", f"userinfo answered {status}")
        email = (info.get("email") or "").strip().lower()
        if not email:
            raise CloudAuthError("Google returned no account email.")
        return {
            "refresh_token": _require_refresh(token, "Google"),
            "account_email": email,
            "account_subject": str(info.get("sub") or "") or None,
        }

    if not settings.onedrive_client_id or not settings.onedrive_client_secret:
        raise AppError(400, "provider_not_configured", "OneDrive storage is not configured on this server.")
    token = _form_post(
        MICROSOFT_OAUTH_TOKEN_URL.format(tenant=settings.onedrive_tenant),
        {
            "client_id": settings.onedrive_client_id,
            "client_secret": settings.onedrive_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "scope": ONEDRIVE_SCOPE,
        },
        "onedrive",
    )
    access = _require_grant(token, "OneDrive")
    status, _, me = http_json("GET", MICROSOFT_GRAPH_ME_URL, headers=_bearer(access))
    if status >= 400:
        raise upstream("onedrive", f"graph /me answered {status}")
    email = (me.get("userPrincipalName") or me.get("mail") or "").strip().lower()
    if not email:
        raise CloudAuthError("OneDrive returned no account email.")
    return {
        "refresh_token": _require_refresh(token, "OneDrive"),
        "account_email": email,
        "account_subject": str(me.get("id") or "") or None,
    }


def refresh_access_token(provider: str, refresh_token: str) -> str:
    """A fresh access token for one provider call."""
    if provider == "google":
        if not settings.google_client_id or not settings.google_client_secret:
            raise AppError(400, "provider_not_configured", "Google Drive storage is not configured.")
        token = _form_post(
            GOOGLE_OAUTH_TOKEN_URL,
            {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            "google",
        )
    else:
        if not settings.onedrive_client_id or not settings.onedrive_client_secret:
            raise AppError(400, "provider_not_configured", "OneDrive storage is not configured.")
        token = _form_post(
            MICROSOFT_OAUTH_TOKEN_URL.format(tenant=settings.onedrive_tenant),
            {
                "client_id": settings.onedrive_client_id,
                "client_secret": settings.onedrive_client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
                "scope": ONEDRIVE_SCOPE,
            },
            "onedrive",
        )
    return _require_grant(token, provider)


def providers_available() -> Dict[str, bool]:
    return {
        "google": bool(settings.google_client_id and settings.google_client_secret),
        "onedrive": bool(settings.onedrive_client_id and settings.onedrive_client_secret),
    }


def _bearer(access_token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


# -------------------------------------------------------------------- upload


@dataclass
class UploadTarget:
    """Where the client uploads the bytes directly, and how.

    Google Drive and OneDrive both answer a resumable upload session URL; the
    client PUTs the raw bytes there with the returned headers.
    """

    upload_url: str
    upload_headers: Dict[str, str]
    expires_at: Optional[str] = None


def create_upload_target(
    provider: str,
    access_token: str,
    remote_name: str,
    mime_type: str,
    size: int,
) -> UploadTarget:
    """Open an upload session for `remote_name` in the user's app folder."""
    if provider == "google":
        url = f"{GOOGLE_UPLOAD_SESSION_URL}?uploadType=resumable&fields=id"
        status, resp_headers, _ = http_bytes(
            "POST",
            url,
            headers={
                **_bearer(access_token),
                "Content-Type": "application/json",
                "X-Upload-Content-Type": mime_type,
                "X-Upload-Content-Length": str(size),
            },
            body=json.dumps({"name": remote_name, "mimeType": mime_type}).encode(),
        )
        if status >= 400:
            raise upstream("google", f"upload session answered {status}")
        upload_url = resp_headers.get("Location") or resp_headers.get("location")
        if not upload_url:
            raise upstream("google", "upload session returned no Location")
        return UploadTarget(
            upload_url=upload_url,
            upload_headers={"Content-Type": mime_type, "Content-Length": str(size)},
        )

    # OneDrive: an upload session inside the app's dedicated folder (approot),
    # so contributor uploads never scatter across the user's own files.
    path = remote_name.replace("'", "''")
    status, _, data = http_json(
        "POST",
        MICROSOFT_GRAPH_UPLOAD_URL.format(path=path),
        headers={**_bearer(access_token), "Content-Type": "application/json"},
        json_body={"item": {"@microsoft.graph.conflictBehavior": "rename"}},
    )
    if status >= 400:
        raise upstream("onedrive", f"upload session answered {status}")
    upload_url = data.get("uploadUrl")
    if not upload_url:
        raise upstream("onedrive", "upload session returned no uploadUrl")
    return UploadTarget(
        upload_url=upload_url,
        upload_headers={"Content-Length": "0"},
        expires_at=data.get("expirationDateTime"),
    )


# ------------------------------------------------------------------ download


@dataclass
class CloudRead:
    """How to stream one provider file. The API re-wraps this in a StreamingResponse."""

    status: int
    headers: Dict[str, str]
    size: int
    body: Iterator[bytes]


def _download_url(provider: str, file_id: str) -> str:
    if provider == "google":
        return GOOGLE_DOWNLOAD_URL.format(file_id=file_id)
    return MICROSOFT_GRAPH_DOWNLOAD_URL.format(file_id=file_id)


def _size_from(headers: Dict[str, str]) -> int:
    content_range = headers.get("Content-Range") or headers.get("content-range") or ""
    if "/" in content_range:
        tail = content_range.rsplit("/", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    length = headers.get("Content-Length") or headers.get("content-length") or ""
    return int(length) if length.isdigit() else 0


def _probe(provider: str, access_token: str, file_id: str) -> Tuple[str, int]:
    """Learn (mime, total size) with a cheap ranged probe."""
    status, headers, body = http_bytes(
        "GET",
        _download_url(provider, file_id),
        headers={**_bearer(access_token), "Range": "bytes=0-0"},
        stream=True,
    )
    if hasattr(body, "close"):
        body.close()
    if status in (401, 403, 404):
        raise AppError(404, "cloud_file_unavailable", "The file is not available in the connected drive.")
    if status >= 400:
        raise upstream(provider, f"download answered {status}")
    mime = headers.get("Content-Type") or headers.get("content-type") or "application/octet-stream"
    return mime, _size_from(headers)


def open_download(
    provider: str,
    access_token: str,
    file_id: str,
    range_header: Optional[str] = None,
) -> CloudRead:
    """Open a (ranged) stream over a stored file.

    `range_header` is the raw `Range: bytes=...` value forwarded to the
    provider, which does the ranging, so a ranged media request comes back a
    real 206 with the right Content-Range. A probe first learns size/mime so a
    headless ranged response still carries Content-Length for the player.
    """
    mime, size = _probe(provider, access_token, file_id)
    headers = _bearer(access_token)
    if range_header:
        headers["Range"] = range_header
    status, resp_headers, body = http_bytes(
        "GET", _download_url(provider, file_id), headers=headers, stream=True
    )
    if status >= 400:
        if hasattr(body, "close"):
            body.close()
        if status in (401, 403, 404):
            raise AppError(404, "cloud_file_unavailable", "The file is not available in the connected drive.")
        raise upstream(provider, f"download answered {status}")
    return CloudRead(
        status=status,
        headers={"Content-Type": resp_headers.get("Content-Type") or mime},
        size=size,
        body=body,
    )
