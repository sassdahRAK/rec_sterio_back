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


# ---------------------------------------------------------------- videos


class VideoCreate(BaseModel):
    source_url: str
    title: Optional[str] = None
    notes: Optional[str] = None
    start_transcription: bool = True


class VideoUpdate(BaseModel):
    title: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_url: str
    platform: str
    title: Optional[str] = None
    notes: Optional[str] = None
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
