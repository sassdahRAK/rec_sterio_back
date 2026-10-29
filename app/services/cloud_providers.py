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
import urllib.parse as _urlparse
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
    "encrypt_token",
    "decrypt_token",
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

# drive.file: only files THIS APP created — cannot read/write the user's other files.
DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"

# Files.ReadWrite.AppFolder: read/write scoped to the app's own isolated folder
# (approot / 'special folder'). Microsoft's approot special folder isolates
# files to this app's namespace so contributor uploads never touch the user's
# other files. Using AppFolder instead of plain Files.ReadWrite is the
# tightest scope available while still supporting resumable upload sessions.
#
# User.Read is separate and required: the token exchange reads the signed-in
# identity from Graph `/me` (userPrincipalName / mail / id) to record which
# account was linked, and Files.ReadWrite.AppFolder alone does NOT grant `/me`
# — without User.Read that call answers 403 and the connect fails after the
# user has already consented. It only permits reading the caller's own basic
# profile; file access stays confined to the app folder.
ONEDRIVE_SCOPE = (
    "offline_access https://graph.microsoft.com/Files.ReadWrite.AppFolder "
    "https://graph.microsoft.com/User.Read"
)

PROVIDER_NAMES = ("google", "onedrive")


# ---------------------------------------------------------- token encryption
#
# Drive refresh tokens are the one secret `drive_links` holds, so at rest they
# are stored under AES-256-GCM from the `cryptography` package (a real AEAD:
# random 12-byte nonce, ciphertext with its auth tag). The envelope is
# `enc1.` + base64url(nonce || ciphertext || tag) — the `enc1.` marker is what
# distinguishes this format from the older one below.
#
# Two legacy shapes can still be on disk:
#   * rows written by the previous hand-rolled SHA-256-CTR + HMAC scheme
#     (no marker) when CLOUD_TOKEN_KEY was set — those still decrypt through
#     `_legacy_decrypt`, which verifies the same HMAC tag;
#   * rows written when CLOUD_TOKEN_KEY was unset — the raw plaintext token.
#     No key makes them readable; they fail both formats' checks and are
#     answered with "reconnect your drive" instead of sending such a token
#     anywhere.

_ENVELOPE_MARKER = "enc1."


def _get_cloud_key() -> Optional[bytes]:
    """The 32-byte AES key decoded from CLOUD_TOKEN_KEY, or None if unset/invalid."""
    raw = (settings.cloud_token_key or "").strip()
    if not raw:
        return None
    import base64
    try:
        key = base64.urlsafe_b64decode(raw + "==")
        if len(key) != 32:
            logger.warning("cloud_token_key must be a 32-byte base64url string; tokens will not be encrypted")
            return None
        return key
    except Exception:
        logger.warning("cloud_token_key is invalid base64; tokens will not be encrypted")
        return None


def encrypt_token(plain: str) -> str:
    """Encrypt a refresh token for storage under AES-256-GCM.

    Refuses when CLOUD_TOKEN_KEY is unset (unless ALLOW_INSECURE=1 marks an
    unsafe-by-design local deployment) — an unencrypted refresh token must
    never be what gets written.
    """
    key = _get_cloud_key()
    if key is None:
        from app.config import _insecure_allowed

        if _insecure_allowed():
            logger.warning("CLOUD_TOKEN_KEY not set — refresh token stored unencrypted (ALLOW_INSECURE)")
            return plain
        raise AppError(
            400,
            "cloud_token_key_required",
            "The server cannot store drive tokens securely: CLOUD_TOKEN_KEY is "
            "not configured. The operator must generate one (see the settings "
            "comment in app/config.py) or set ALLOW_INSECURE=1 for local "
            "development only.",
        )
    import base64
    import os

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plain.encode("utf-8"), None)
    payload = nonce + ciphertext  # ciphertext carries its 16-byte auth tag
    return _ENVELOPE_MARKER + base64.urlsafe_b64encode(payload).decode()


def _legacy_decrypt(stored: str) -> Optional[str]:
    """The old SHA-256-CTR + HMAC envelope, still readable for migration-free upgrade.

    Returns None when the value is not in this envelope (bad base64, HMAC
    mismatch, wrong length) — the caller then treats it as unusable data.
    """
    import base64
    import hashlib
    import hmac
    import struct

    try:
        payload = base64.urlsafe_b64decode(stored + "==")
        if len(payload) < 12 + 16:
            return None
        nonce = payload[:12]
        tag = payload[-16:]
        ciphertext = payload[12:-16]
    except Exception:  # noqa: BLE001 — not our envelope at all
        return None
    key = _get_cloud_key()
    if key is None:
        return None
    aes_key = hashlib.sha256(b"aes:" + key).digest()
    mac_key = hashlib.sha256(b"mac:" + key).digest()
    expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(tag, expected_tag):
        return None
    keystream = b""
    counter = 0
    while len(keystream) < len(ciphertext):
        keystream += hashlib.sha256(aes_key + nonce + struct.pack(">Q", counter)).digest()
        counter += 1
    plain = bytes(a ^ b for a, b in zip(ciphertext, keystream[: len(ciphertext)]))
    try:
        return plain.decode("utf-8")
    except UnicodeDecodeError:
        return None


def decrypt_token(stored: str) -> str:
    """Decrypt a stored refresh token, or refuse.

    With ALLOW_INSECURE=1 and no key configured (local dev), the stored value
    passes through unchanged. Otherwise a value that is not a valid envelope
    is refused — in particular the rows that were stored as raw plaintext
    before any key existed can never be honoured silently.
    """
    from app.config import _insecure_allowed

    key = _get_cloud_key()
    if key is None:
        if _insecure_allowed():
            return stored
        raise AppError(
            401,
            "cloud_auth_rejected",
            "The server can no longer read this drive's tokens: CLOUD_TOKEN_KEY "
            "is not configured. Reconnect your drive to restore access.",
        )
    raw = stored or ""
    if raw.startswith(_ENVELOPE_MARKER):
        import base64

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        try:
            payload = base64.urlsafe_b64decode(raw[len(_ENVELOPE_MARKER):] + "==")
            if len(payload) < 12 + 16:
                raise ValueError("too short")
            nonce, ciphertext = payload[:12], payload[12:]
            plain = AESGCM(key).decrypt(nonce, ciphertext, None)
            return plain.decode("utf-8")
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001 — any envelope failure is "reconnect"
            raise AppError(
                401,
                "cloud_auth_rejected",
                "Drive token integrity check failed — reconnect your drive.",
            ) from exc
    legacy = _legacy_decrypt(raw)
    if legacy is not None:
        return legacy
    # Neither envelope: a raw-plaintext-at-rest row (key never set when it was
    # written) or corrupt data. Refuse rather than send token-like garbage to
    # a provider.
    raise AppError(
        401,
        "cloud_auth_rejected",
        "This drive's stored token cannot be decrypted — reconnect your drive.",
    )


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
        g_id = settings.drive_google_client_id or settings.google_client_id
        g_secret = settings.drive_google_client_secret or settings.google_client_secret
        if not g_id or not g_secret:
            raise AppError(400, "provider_not_configured", "Google Drive storage is not configured on this server.")
        token = _form_post(
            GOOGLE_OAUTH_TOKEN_URL,
            {
                "client_id": g_id,
                "client_secret": g_secret,
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
        g_id = settings.drive_google_client_id or settings.google_client_id
        g_secret = settings.drive_google_client_secret or settings.google_client_secret
        if not g_id or not g_secret:
            raise AppError(400, "provider_not_configured", "Google Drive storage is not configured.")
        token = _form_post(
            GOOGLE_OAUTH_TOKEN_URL,
            {
                "client_id": g_id,
                "client_secret": g_secret,
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
    g_id = settings.drive_google_client_id or settings.google_client_id
    g_secret = settings.drive_google_client_secret or settings.google_client_secret
    return {
        "google": bool(g_id and g_secret),
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
    # Encode the path for the Graph URL (spaces → %20, etc.)
    encoded_path = _urlparse.quote(remote_name, safe='')
    status, _, data = http_json(
        "POST",
        MICROSOFT_GRAPH_UPLOAD_URL.format(path=encoded_path),
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
        upload_headers={},   # client sets Content-Length and Content-Range for each chunk
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
