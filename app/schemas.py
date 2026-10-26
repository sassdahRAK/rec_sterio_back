"""Pydantic v2 request/response models. snake_case everywhere."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator


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


# ---------------------------------------------------------------- videos


class VideoCreate(BaseModel):
    source_url: str
    title: Optional[str] = None
    notes: Optional[str] = None
    start_transcription: bool = True
    # Where the ingestion lands; NULL keeps it in the unfiled library.
    folder_id: Optional[UUID] = None


class VideoUpdate(BaseModel):
    title: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    folder_id: Optional[UUID] = None
    position: Optional[int] = None


class VideoMove(BaseModel):
    folder_id: Optional[UUID] = None
    position: Optional[int] = None


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_url: str
    platform: str
    title: Optional[str] = None
    notes: Optional[str] = None
    folder_id: Optional[UUID] = None
    position: int = 0
    status: str
    progress: int
    error: Optional[str] = None
    local_path: Optional[str] = None
    file_size: Optional[int] = None
    duration_seconds: Optional[float] = None
    mime_type: Optional[str] = None
    checksum_sha256: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_by: str
    created_at: UTCDateTime
    updated_at: UTCDateTime


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


class VideoDetailOut(VideoOut):
    transcripts: List[TranscriptOut] = Field(default_factory=list)
    jobs: List[JobOut] = Field(default_factory=list)


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
    folder_id: UUID
    code: str = Field(min_length=1, max_length=64)
    device_key: Optional[str] = None


class ShareRedeem(BaseModel):
    device_key: Optional[str] = None


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
    duration_seconds: Optional[float] = None
    thumbnail_url: Optional[str] = None
    folder_id: Optional[UUID] = None
    folder_name: Optional[str] = None
    badge: Literal["raw", "cleaned"]
    transcript_id: Optional[UUID] = None
    transcript_preview: Optional[str] = None
    audio_available: bool = False
    media_url: str
    audio_url: str
    transcript_url: str


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
