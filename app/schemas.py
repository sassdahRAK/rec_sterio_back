"""Pydantic v2 request/response models. snake_case everywhere."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    computed_field,
    field_validator,
)

from app.roles import ACCOUNT_ROLES, ROLE_USER


def _utc_iso(value: datetime) -> str:
    """ISO-8601 UTC with a Z suffix (contract requirement)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# All timestamps in the API are UTC / Z.
UTCDateTime = Annotated[
    datetime, PlainSerializer(_utc_iso, return_type=str, when_used="json")
]

# ---------------------------------------------------------------- segments

# Technical error text (ffmpeg stderr, Python tracebacks of a failed step) is
# useful to an operator but must not become a filesystem/machine oracle at the
# API boundary: absolute paths, argv fragments and multi-line stderr dumps are
# reduced to a one-line, path-free reason.
_WIN_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']*[\\/][^\s\"']*[^\s\"'.,;:)!}?]*")
_UNIX_PATH_RE = re.compile(r"(?<![:\w])/[^\s\"']*/[^\s\"']*[^\s\"'.,;:)!}?]*")


def sanitize_technical_error(value: Optional[str], limit: int = 300) -> Optional[str]:
    """One line, no absolute paths, bounded length; short human reasons pass.

    Applied where a `error`/`last_error`-style field leaves the schema layer:
    the stored value may contain a path and stderr lines (they help the
    operator debug), while responses keep a readable reason only.
    """
    if value is None:
        return None
    first = next((ln.strip() for ln in str(value).splitlines() if ln.strip()), "")
    line = " ".join(first.split()) or str(value).strip()
    line = _WIN_PATH_RE.sub("[path]", line)
    line = _UNIX_PATH_RE.sub("[path]", line)
    if len(line) > limit:
        line = line[: limit - 1].rstrip() + "…"
    return line or None


class Segment(BaseModel):
    model_config = ConfigDict(extra="allow")

    start: float = 0.0
    end: float = 0.0
    text: str = ""
    language: Optional[str] = None


class SegmentList(BaseModel):
    segments: List[Segment] = Field(default_factory=list)


# ---------------------------------------------------------------- auth


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()


class LoginResponse(BaseModel):
    token: str
    email: str
    expires_at: UTCDateTime
    role: str = "admin"
    name: Optional[str] = None


class PrincipalOut(BaseModel):
    """Who the caller is, as the API sees them."""

    actor: str
    role: str
    email: Optional[str] = None
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    folder_ids: List[UUID] = Field(default_factory=list)


class GuestTokenOut(BaseModel):
    token: str
    role: str = "guest"
    email: Optional[str] = None
    folder_ids: List[UUID] = Field(default_factory=list)
    expires_at: UTCDateTime


class GoogleCredentialRequest(BaseModel):
    """The ID token Google Identity Services hands the browser."""

    credential: str


class TelegramLinkOut(BaseModel):
    """A freshly generated link code, shown to the console user exactly once.

    Only the sha256 hash is stored; this response is the one place the
    plaintext code ever appears.
    """

    code: str
    # https://t.me/<bot>?start=link-<code> when the bot username is configured;
    # None in dev, where the console shows the bare code to type instead.
    deep_link: Optional[str] = None
    expires_at: UTCDateTime


class TelegramLinkStatusOut(BaseModel):
    """The caller's own Telegram link, or null when unlinked."""

    telegram_user_id: Optional[int] = None


# ---------------------------------------------------------------- accounts
#
# The user-management surface (`routers/users.py`, super admin only). The rules
# that can be expressed as a field live here — email shape, the role list from
# `app.roles`, the password floor — so the route itself only has to handle the
# two things a schema cannot: the duplicate-email race and self-modification.

# Eight characters: the floor the console's create dialog enforces, so the two
# cannot disagree about what is acceptable.
MIN_PASSWORD_LENGTH = 8


def _normalized_email(value: str) -> str:
    """Strip and lower-case, so case alone cannot create a second account.

    A full RFC check is not worth its false negatives; `@` with something on
    both sides is the minimum that keeps the account reachable by its owner.
    """
    email = (value or "").strip().lower()
    if "@" not in email or " " in email or email.startswith("@") or email.endswith("@"):
        raise ValueError("A valid email address is required.")
    return email


def _known_role(value: str) -> str:
    if value not in ACCOUNT_ROLES:
        raise ValueError("role must be one of: " + ", ".join(ACCOUNT_ROLES) + ".")
    return value


class AdminUserOut(BaseModel):
    """One console account. Never carries the password hash."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    name: Optional[str] = None
    role: str
    is_active: bool
    telegram_user_id: Optional[int] = None
    created_at: UTCDateTime


class AdminUserListOut(BaseModel):
    items: List[AdminUserOut]
    total: int
    page: int
    page_size: int


class UserCreateIn(BaseModel):
    email: str
    name: Optional[str] = None
    role: str = ROLE_USER
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        return _normalized_email(value)

    @field_validator("role")
    @classmethod
    def _role(cls, value: str) -> str:
        return _known_role(value)


class UserPatchIn(BaseModel):
    """Everything a super admin may change except the identity itself.

    `email` is deliberately absent: it is the account key (`Principal.email`
    resolves the row and the actor strings embed it), so renaming one would
    orphan ownership of every folder, membership and transcript it authored.
    Every field is optional because this is a PATCH — the route reads
    `exclude_unset` to tell "leave it alone" from "set it to null".
    """

    role: Optional[str] = None
    is_active: Optional[bool] = None
    name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=MIN_PASSWORD_LENGTH)

    @field_validator("role")
    @classmethod
    def _role(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _known_role(value)


class UserDeleteIn(BaseModel):
    """Confirmation the account holder supplies with `DELETE /users/me`.

    Deleting yourself while still owning folders takes those folders out of
    management, so an account with owned folders must send `{"confirm": true}`
    to have them handed to the seeded operator first. Without it the deletion
    is refused with a 409.
    """

    confirm: bool = False


class DriveLinkOut(BaseModel):
    """One connected personal cloud. Never carries the refresh token."""

    id: UUID
    provider: str  # 'google' | 'onedrive'
    account_email: str
    connected_at: UTCDateTime


class DriveProviderStatus(BaseModel):
    """Which providers this deployment offers, for the connect UI."""

    google: bool
    onedrive: bool
    # OAuth client ids are public by design — they appear in the browser's
    # consent URL — so publishing them lets the portal build that URL without
    # a rebuild. `None` means this provider is not configured on this server.
    google_client_id: Optional[str] = None
    onedrive_client_id: Optional[str] = None


class DriveConnectIn(BaseModel):
    """The OAuth authorization code the provider handed back to the client.

    `redirect_uri` must be the same value the browser sent to the provider's
    consent screen, or the token exchange is rejected. It is optional so a
    client can rely on the server's configured default, and the route accepts
    it only when its origin matches the configured portal origin.
    """

    provider: str
    code: str = Field(min_length=1)
    redirect_uri: Optional[str] = None

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        if value not in ("google", "onedrive"):
            raise ValueError("provider must be 'google' or 'onedrive'.")
        return value


class CloudUploadIn(BaseModel):
    """Store this upload on the contributor's own cloud instead of server disk."""

    backend: str
    filename: Optional[str] = Field(None, max_length=255)
    size: Optional[int] = Field(None, ge=1)

    @field_validator("backend")
    @classmethod
    def _backend(cls, value: str) -> str:
        if value not in ("google", "onedrive"):
            raise ValueError("backend must be 'google' or 'onedrive'.")
        return value


class CloudUploadOut(BaseModel):
    video_id: UUID
    upload_url: str
    upload_headers: Dict[str, str]
    expires_at: Optional[UTCDateTime] = None


class CompleteCloudUploadIn(BaseModel):
    """The provider file id for the uploaded video, plus optional client-measured metadata."""

    provider_file_id: str = Field(min_length=1)
    size: Optional[int] = Field(None, ge=1)
    checksum_sha256: Optional[str] = None


# ---------------------------------------------------------------- videos


class VideoCreate(BaseModel):
    source_url: str
    title: Optional[str] = None
    notes: Optional[str] = None
    start_transcription: bool = True
    # Where the ingestion lands; NULL keeps it in the unfiled library.
    folder_id: Optional[UUID] = None


class VideoUpdate(BaseModel):
    """The fields an editor may change on a video row.

    `status` is deliberately NOT editor-writable: the processing lifecycle is
    owned by the queue/runner and the ingestion routes, and a hand-poked
    status (a "failed" row claimed transcribed) breaks the pipeline's
    assumptions. A caller who needs a stuck row moving again uses
    `POST /videos/{id}/retry`, which resets it properly.
    """

    title: Optional[str] = None
    notes: Optional[str] = None
    folder_id: Optional[UUID] = None
    position: Optional[int] = None


class VideoMove(BaseModel):
    folder_id: Optional[UUID] = None
    position: Optional[int] = None


# ---------------------------------------------------------------- TvNa record
# instruction.txt #6: the collection schema the TvNa platform requires. One
# record per collected item, attached to the video it describes.

SOURCE_TYPES = ("Video", "Discover Page", "Profile")
COLLECTION_METHODS = ("Manual browse", "Research API")


class TvnaRecordIn(BaseModel):
    """The fields a collector supplies. `location_tag` is the only required one.

    Everything else either has a sane default or is filled in from the download
    metadata (platform, author, date, caption) before the collector is asked.
    """

    location_tag: str = Field(min_length=1, max_length=500)
    source_type: Literal["Video", "Discover Page", "Profile"] = "Video"
    author_handle: Optional[str] = Field(None, max_length=255)
    date_posted: Optional[str] = Field(None, max_length=16)
    caption: Optional[str] = None
    collection_method: Literal["Manual browse", "Research API"] = "Manual browse"
    collected_by: Optional[str] = Field(None, max_length=128)
    collection_date: Optional[str] = Field(None, max_length=16)
    notes: Optional[str] = None


class TelegramLinkIn(BaseModel):
    """The Telegram copy of a video (instruction.txt #4).

    `file_id` is what `getFile` needs; `message_id` locates the message it
    arrived in, which is what makes it findable again in the chat.
    """

    file_id: str = Field(min_length=1, max_length=255)
    message_id: Optional[int] = Field(None, ge=0)


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_url: str
    platform: str
    title: Optional[str] = None
    notes: Optional[str] = None
    folder_id: Optional[UUID] = None
    # Set when this row is a fork of another video, NULL for an ingested one.
    copied_from_id: Optional[UUID] = None
    position: int = 0
    status: str
    progress: int
    error: Optional[str] = None
    # `local_path` is deliberately NOT exposed: the stored column names the
    # server's media directory, which is not something an API response should
    # disclose even basename-only. Clients stream through /videos/{id}/media.
    file_size: Optional[int] = None
    duration_seconds: Optional[float] = None
    mime_type: Optional[str] = None
    checksum_sha256: Optional[str] = None
    thumbnail_url: Optional[str] = None

    # Personal-cloud storage (the contributor's own Google Drive / OneDrive), so
    # the detail views can say where the bytes live.
    storage_backend: Optional[str] = None

    # Telegram mirror (instruction.txt #4).
    telegram_file_id: Optional[str] = None
    telegram_message_id: Optional[int] = None

    # TvNa collection schema (instruction.txt #6).
    tvna_serial: Optional[int] = None
    source_type: Optional[str] = None
    author_handle: Optional[str] = None
    date_posted: Optional[str] = None
    caption: Optional[str] = None
    location_tag: Optional[str] = None
    collection_method: Optional[str] = None
    collected_by: Optional[str] = None
    collection_date: Optional[str] = None

    created_by: str
    created_at: UTCDateTime
    updated_at: UTCDateTime

    # Whether THIS caller could open the video (pass `GET /videos/{id}`'s
    # `require_video`). The listing routes set it per row for consumers; the
    # library listings now hide rows a caller cannot open at all, so it is
    # True for every listed row. Defaults to True: every other VideoOut
    # response belongs to a caller who has just passed an editor/read check
    # on exactly this video.
    can_open: bool = True

    @field_validator("error")
    @classmethod
    def _sanitize_error(cls, v: Optional[str]) -> Optional[str]:
        """The stored value may carry absolute paths, argv and multi-line
        stderr from a failed step; the response keeps a one-line, path-free
        human reason. Inherited by VideoDetailOut (and list rows), so every
        video error an API caller sees passes through here."""
        return sanitize_technical_error(v)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tvna_id(self) -> Optional[str]:
        """The public TvNa identifier, e.g. "PP-001".

        Derived from the integer serial so the format lives in exactly one
        place — nothing in the database stores a "PP-" string.
        """
        return f"PP-{self.tvna_serial:03d}" if self.tvna_serial else None


class TranscriptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    video_id: UUID
    version: int
    source: str
    language: Optional[str] = None
    content: str
    segments: Optional[List[Dict[str, Any]]] = None
    status: str
    # instruction.txt #8: whether a human has corrected this version (Cleaned
    # card) or it is still the raw ASR output (Raw card).
    cleaned: bool = False
    error: Optional[str] = None
    created_by: str
    created_at: UTCDateTime
    updated_at: UTCDateTime

    @field_validator("error")
    @classmethod
    def _sanitize_error(cls, v: Optional[str]) -> Optional[str]:
        """Same boundary rule as `VideoOut.error`: the stored step error may
        carry paths and stderr lines; `GET /videos/{id}` embeds transcripts
        (`VideoDetailOut`), so a failed version must not leak them either."""
        return sanitize_technical_error(v)


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kind: str
    video_id: Optional[UUID] = None
    status: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    attempts: int
    max_attempts: int
    priority: int
    last_error: Optional[str] = None
    scheduled_at: UTCDateTime
    started_at: Optional[UTCDateTime] = None
    finished_at: Optional[UTCDateTime] = None
    created_at: UTCDateTime

    @field_validator("last_error")
    @classmethod
    def _sanitize_last_error(cls, v: Optional[str]) -> Optional[str]:
        """Same boundary rule as `VideoOut.error`: `GET /videos/{id}` embeds
        jobs (`VideoDetailOut`) and the stored value carries worker stderr,
        argv and file paths — the response keeps a one-line human reason."""
        return sanitize_technical_error(v)


class VideoDetailOut(VideoOut):
    transcripts: List[TranscriptOut] = Field(default_factory=list)
    jobs: List[JobOut] = Field(default_factory=list)
    # Readable names for the actor strings, resolved only on the detail view
    # (a linked Telegram user shows as "Name (email)"). The raw `created_by` /
    # `collected_by` stay untouched above: they are the authorization source of
    # truth, and None simply means "no readable name" — the caller shows the
    # raw string it already has.
    created_by_display: Optional[str] = None
    collected_by_display: Optional[str] = None


class VideoListOut(BaseModel):
    items: List[VideoOut]
    total: int
    page: int
    page_size: int


class JobListOut(BaseModel):
    items: List[JobOut]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------- transcripts


class TranscriptUpdate(BaseModel):
    content: str
    segments: Optional[List[Dict[str, Any]]] = None
    language: Optional[str] = None
    # None means "infer": a manual edit is a human correction, so it is cleaned.
    cleaned: Optional[bool] = None


class TranscriptCleanedUpdate(BaseModel):
    """Toggle the Raw/Cleaned badge on the current transcript version (#8)."""

    cleaned: bool


class TranscriptListOut(BaseModel):
    items: List[TranscriptOut]


class GenerateRequest(BaseModel):
    engine: Literal["whisper", "gemini", "both"] = "whisper"


class GenerateResponse(BaseModel):
    jobs: List[JobOut]


# ---------------------------------------------------------------- merges


class MergeCreate(BaseModel):
    base_transcript_id: Optional[UUID] = None
    gemini_transcript_id: Optional[UUID] = None
    model: Optional[str] = None


class MergeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    video_id: UUID
    base_transcript_id: Optional[UUID] = None
    gemini_transcript_id: Optional[UUID] = None
    merged_transcript_id: Optional[UUID] = None
    status: str
    diff: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    error: Optional[str] = None
    created_at: UTCDateTime
    updated_at: UTCDateTime

    @field_validator("error")
    @classmethod
    def _sanitize_error(cls, v: Optional[str]) -> Optional[str]:
        """Same boundary rule as `VideoOut.error`: the worker stores
        `str(exc)[:2000]`, which can carry a path or ffmpeg stderr; the
        response keeps a one-line, path-free reason."""
        return sanitize_technical_error(v)


class MergeListOut(BaseModel):
    items: List[MergeOut]


# ---------------------------------------------------------------- stats / health


class HealthOut(BaseModel):
    status: str
    db: bool
    worker_alive: bool


class StatsOut(BaseModel):
    videos_total: int
    videos_by_status: Dict[str, int]
    transcripts_total: int
    transcripts_by_source: Dict[str, int]
    merges_total: int
    jobs_queued: int
    jobs_failed: int
    total_media_bytes: int


# ---------------------------------------------------------------- folders
# Dataset sharing (instruction.txt #1, #4, #6, #8, #9). A folder groups videos
# into a publishable unit; sub-folders mirror the ZIP export layout.

class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    visibility: Literal["public", "private"] = "private"
    parent_id: Optional[UUID] = None
    position: int = 0


class FolderUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    visibility: Optional[Literal["public", "private"]] = None
    parent_id: Optional[UUID] = None
    position: Optional[int] = None


class FolderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: Optional[str] = None
    visibility: str
    owner_id: UUID
    owner_email: Optional[str] = None
    parent_id: Optional[UUID] = None
    position: int
    created_at: UTCDateTime
    updated_at: UTCDateTime
    video_count: int = 0
    # The caller's effective role: owner | editor | commenter | viewer.
    my_role: Optional[str] = None
    children: List["FolderOut"] = Field(default_factory=list)


class FolderListOut(BaseModel):
    items: List[FolderOut]
    total: int


class VideoFolderUpdate(BaseModel):
    """Place a video in a folder (and optionally reorder it within that folder)."""

    folder_id: Optional[UUID] = None
    position: Optional[int] = None


class VideoCopyIn(BaseModel):
    """Fork a video into a folder the caller manages, leaving the original alone."""

    folder_id: UUID
    position: Optional[int] = None


# ---------------------------------------------------------------- members


class MemberInvite(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    role: Literal["viewer", "editor", "commenter"] = "viewer"


class MemberUpdate(BaseModel):
    role: Literal["viewer", "editor", "commenter"]


class FolderMemberOut(BaseModel):
    user_id: UUID
    email: str
    name: Optional[str] = None
    role: str
    created_at: UTCDateTime


class FolderMemberListOut(BaseModel):
    items: List[FolderMemberOut]


# ---------------------------------------------------------------- share links

class ShareLinkCreate(BaseModel):
    role: Literal["viewer", "editor", "commenter"] = "viewer"
    expires_at: Optional[datetime] = None


class ShareLinkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    folder_id: UUID
    token: str
    role: str
    expires_at: Optional[UTCDateTime] = None
    revoked_at: Optional[UTCDateTime] = None
    created_by: str
    created_at: UTCDateTime


class ShareLinkCreated(ShareLinkOut):
    """The one response that exposes the bare token and a ready-to-send URL."""

    url: str


# ---------------------------------------------------------------- passcodes

class PasscodeCreate(BaseModel):
    """A door code for a private folder (instruction.txt #4, #7).

    `expires_at=None` means "never expires"; `max_uses=None` means unlimited;
    `max_devices=None` means no device cap; `one_time=True` burns the code on its
    first successful redemption. Omit `code` to have one generated (#7).
    """

    code: Optional[str] = Field(None, min_length=4, max_length=64)
    label: Optional[str] = None
    expires_at: Optional[datetime] = None
    max_uses: Optional[int] = Field(None, ge=1)
    max_devices: Optional[int] = Field(None, ge=1)
    one_time: bool = False


class PasscodeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    folder_id: UUID
    label: Optional[str] = None
    expires_at: Optional[UTCDateTime] = None
    max_uses: Optional[int] = None
    max_devices: Optional[int] = None
    one_time: bool
    use_count: int
    device_count: int = 0
    revoked_at: Optional[UTCDateTime] = None
    created_by: str
    created_at: UTCDateTime
    # Returned exactly once, at creation time; the hash is never exposed.
    code: Optional[str] = None


class PasscodeListOut(BaseModel):
    items: List[PasscodeOut]


class PasscodeRedeem(BaseModel):
    """Which folder a code opens.

    `folder_id` comes from a link; `folder_name` is typed by the visitor. Either
    may be omitted — the code alone is enough. The folder list is never exposed,
    so a visitor cannot learn how many folders exist or what they are called.
    """

    folder_id: Optional[UUID] = None
    folder_name: Optional[str] = Field(None, min_length=1, max_length=255)
    code: str = Field(min_length=1, max_length=64)
    device_key: Optional[str] = None


class ShareRedeem(BaseModel):
    device_key: Optional[str] = None


class PortalLinkOut(BaseModel):
    """The signed single-video link the bot appends to its portal button.

    The token is the ordinary app JWT (role "guest", scoped to one video), so
    the portal sends it exactly as it sends any other bearer token; `path` is
    the ready-to-append URL so the bot never assembles the query string itself.
    """

    token: str
    expires_at: UTCDateTime
    path: str


class GuestTokenResponse(BaseModel):
    """A scoped bearer token plus the human-readable way it was obtained."""

    token: str
    expires_at: UTCDateTime
    folder_id: UUID
    # The session role is always "guest"; `folder_role` is the permission the
    # passcode/share actually granted (viewer < commenter < editor), which the
    # portal needs in order to show or hide editing controls.
    role: str
    folder_role: str = "viewer"
    access: Literal["passcode", "share", "public"]
    folder: Optional["FolderOut"] = None


# ---------------------------------------------------------------- public surface

class DataCardOut(BaseModel):
    """One video as the public dataset cards show it (instruction.txt #8, #9)."""

    video_id: UUID
    title: Optional[str] = None
    source_url: Optional[str] = None
    status: str
    # Live download/transcribe progress (0–100) and the last failure message.
    # The portal's per-video page shows a unit before it is finished, so it
    # needs the same two numbers the console's video object carries.
    progress: int = 0
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    thumbnail_url: Optional[str] = None
    folder_id: Optional[UUID] = None
    folder_name: Optional[str] = None
    # A fork of another unit. The card says so instead of passing the copy off
    # as an independently collected item.
    copied_from_id: Optional[UUID] = None
    badge: Literal["raw", "cleaned"]
    transcript_id: Optional[UUID] = None
    transcript_preview: Optional[str] = None
    audio_available: bool = False
    media_url: str
    audio_url: str
    transcript_url: str
    # Whether THIS caller could open the card's video (the card dialog loads
    # its transcript via `GET /videos/{id}`). The dataset listing sets it per
    # card for consumers; it now hides cards a caller cannot open at all, so
    # it is True for every card it returns. Defaults to True: every other
    # builder — the public portal — serves cards pre-filtered to folders the
    # caller has already passed `require_read` on.
    can_open: bool = True
    # Personal-cloud storage: where the bytes live, and whether they can be
    # read right now (a revoked/disconnected drive leaves the unit listed but
    # unavailable, rather than failing playback with a 500).
    storage_backend: Optional[str] = None
    storage_available: bool = True


class PublicFolderOut(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    visibility: str
    # The caller's effective role on this folder, so the portal can show edit
    # controls only to editors and never merely guess from the session.
    my_role: Optional[str] = None
    cards: List[DataCardOut] = Field(default_factory=list)
    children: List["PublicFolderOut"] = Field(default_factory=list)


class PublicFolderListOut(BaseModel):
    items: List[PublicFolderOut]
    total: int


class DatasetCardListOut(BaseModel):
    items: List[DataCardOut]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------- comments

class FolderCommentCreate(BaseModel):
    body: str = Field(min_length=1, max_length=4000)


class FolderCommentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    folder_id: UUID
    # Null when the comment came from a guest with no account.
    author_id: Optional[UUID] = None
    author_label: str
    body: str
    created_at: UTCDateTime


class FolderCommentListOut(BaseModel):
    items: List[FolderCommentOut]
