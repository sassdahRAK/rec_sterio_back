"""Video routes: list, create, detail, update, delete, media streaming, retry."""
from __future__ import annotations

import copy
import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from app.access import (
    folder_or_404,
    require_manage,
    require_read,
    require_video,
    video_role,
    visible_folders,
)
from app.auth import Principal, get_actor, get_optional_principal, get_principal
from app.config import settings
from app.db import get_db
from app.errors import AppError
from app.models import DriveLink, Folder, Job, Merge, Transcript, Video
from app.routers.bot_users import require_session_consent
from app.schemas import (
    JobOut,
    TranscriptOut,
    VideoCopyIn,
    VideoCreate,
    VideoDetailOut,
    VideoListOut,
    VideoOut,
    VideoUpdate,
    TelegramLinkIn,
    TvnaRecordIn,
)
from app.roles import ADMIN_TIER, LIBRARY_VIEW_ROLES
from app.security import detect_platform, enforce_rate_limit, validate_source_url
from app.services import asr, cloud_providers, storage, zip_export
from app.services.cloud_providers import decrypt_token
from app.services.actors import display_name
from app.worker import queue

logger = logging.getLogger("recsterio.videos")

router = APIRouter(prefix="/videos", tags=["videos"])

SORTABLE = {"created_at", "-created_at", "title", "-title"}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _dedupe_scope(db: Session, principal: Principal):
    """The rows an ingest collision may be reported against, or None for all.

    De-duplication used to be global, so submitting a URL that lived in someone
    else's private folder answered `409` with that video's title, notes and id.
    That is a disclosure, not idempotency. Matching only what the caller can
    already read keeps the useful half — their own re-submissions and public
    units — and says nothing about rows they were never allowed to see.

    It also removes the reason a copy had to be special-cased: a fork of a
    public card lives in a private folder, so it is invisible to everyone but
    its owner and cannot shadow the original for anyone else.

    The Telegram service identity (`telegram:bot`) keeps the global view — it
    is the worker/operator actor. A *person* acting through the bot
    (`telegram:<uid>`) is scoped like the console account they linked, so a
    teammate re-pasting a link gets back the unit they can already open, and
    an unrelated user learns nothing about a private folder's contents.
    """
    if principal.role == "telegram" and principal.actor == "telegram:bot":
        return None
    if principal.role == "telegram":
        return _telegram_person_scope(db, principal)
    if principal.role in LIBRARY_VIEW_ROLES:
        return None
    visible = list(visible_folders(db, principal).keys())
    conditions = [Video.created_by == principal.actor]
    if visible:
        conditions.append(Video.folder_id.in_(visible))
    return or_(*conditions)


def _telegram_person_scope(db: Session, principal: Principal):
    """What a person acting through the bot may collide with.

    Their linked console account's visibility (public folders + the folders
    they own or are a member of) plus the rows they ingested through the bot.
    An unlinked Telegram user is limited to public folders and their own rows —
    a link that lives in someone's private folder is invisible to them, and a
    re-paste creates their own copy rather than disclosing the original.
    """
    from app.models import AdminUser

    conditions = [Video.created_by == principal.actor]
    uid = principal.email
    linked = None
    if uid and uid.isdigit():
        linked = db.execute(
            select(AdminUser).where(AdminUser.telegram_user_id == int(uid))
        ).scalar_one_or_none()
    if linked is not None:
        as_console = Principal(f"{linked.role}:{linked.email}", linked.role, linked.email)
        visible = list(visible_folders(db, as_console).keys())
        conditions.append(Video.created_by == as_console.actor)
        if visible:
            conditions.append(Video.folder_id.in_(visible))
    else:
        public = [
            f.id for f in db.execute(select(Folder).where(Folder.visibility == "public")).scalars()
        ]
        if public:
            conditions.append(Video.folder_id.in_(public))
    return or_(*conditions)


def _lock_on_source_url(db: Session, source_url: str) -> None:
    """Serialise concurrent ingests of one URL for the rest of this transaction.

    Released automatically on commit or rollback, so it cannot leak. Only
    PostgreSQL has advisory locks; elsewhere the de-duplication check keeps its
    old best-effort behaviour. Mirrors the TvNa serial allocation.
    """
    if db.get_bind().dialect.name != "postgresql":
        return
    db.execute(
        text("select pg_advisory_xact_lock(hashtextextended(:url, 0))"),
        {"url": source_url},
    )


def get_video_or_404(db: Session, video_id: UUID) -> Video:
    video = db.get(Video, video_id)
    if video is None:
        raise AppError(404, "not_found", "Video not found.", [{"id": str(video_id)}])
    return video


@router.get("", response_model=VideoListOut)
def list_videos(
    page: int = Query(1, ge=1),
    # 1000 lets the console's "All" page size actually return the library; the
    # cap still bounds a single response.
    page_size: int = Query(20, ge=1, le=1000),
    status: Optional[str] = None,
    platform: Optional[str] = None,
    q: Optional[str] = None,
    folder_id: Optional[UUID] = None,
    sort: str = Query("-created_at"),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoListOut:
    stmt = select(Video)
    count_stmt = select(func.count()).select_from(Video)

    # The console's library is only fully visible to the operator tier and the
    # reviewer. A member sees the folders shared with them (plus what they
    # ingested themselves), so a folder's privacy is not merely cosmetic. The
    # library tiers' rows are additionally dropped from the page when the
    # caller could not open them — see the pagination branch below.
    if principal.role not in LIBRARY_VIEW_ROLES + ("telegram",):
        roles = visible_folders(db, principal)
        conditions = [Video.created_by == principal.actor]
        if roles:
            conditions.append(Video.folder_id.in_(list(roles.keys())))
        stmt = stmt.where(or_(*conditions))
        count_stmt = count_stmt.where(or_(*conditions))
    if folder_id is not None:
        stmt = stmt.where(Video.folder_id == folder_id)
        count_stmt = count_stmt.where(Video.folder_id == folder_id)
    if status:
        stmt = stmt.where(Video.status == status)
        count_stmt = count_stmt.where(Video.status == status)
    if platform:
        stmt = stmt.where(Video.platform == platform)
        count_stmt = count_stmt.where(Video.platform == platform)
    if q:
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(or_(Video.title.ilike(pattern), Video.source_url.ilike(pattern)))
        count_stmt = count_stmt.where(
            or_(Video.title.ilike(pattern), Video.source_url.ilike(pattern))
        )
    if sort not in SORTABLE:
        raise AppError(
            422,
            "validation_failed",
            "sort must be one of created_at, -created_at, title.",
            [{"field": "sort", "issue": "unsupported_value", "allowed": sorted(SORTABLE)}],
        )
    column = {
        "created_at": Video.created_at,
        "-created_at": Video.created_at.desc(),
        "title": Video.title,
        "-title": Video.title.desc(),
    }[sort]
    # `can_open` per row is the same check `GET /videos/{id}` runs
    # (`require_video`), so a row kept in this listing can never 403 on the
    # detail view. The admin tier and the bot pass it unconditionally, so their
    # page keeps being counted and sliced by the database. Everyone else can
    # hold rows the check rules out — the library tiers (developer/reviewer)
    # see private folders they hold no role on — and the decision is that such
    # rows are not listed at all rather than shown unopenable. The check cannot
    # run in SQL (it walks folder chains and memberships), so those callers get
    # the page assembled in Python: filter first, then slice, so `total` counts
    # exactly the rows the caller could open and a page never underfills. A
    # plain user's scope already agrees with the check, so for them this only
    # removes a row should the two ever drift. The lists are small, so one pass
    # over them is fine.
    if principal.role in ADMIN_TIER + ("telegram",):
        total = db.execute(count_stmt).scalar_one()
        items = db.execute(
            stmt.offset((page - 1) * page_size).limit(page_size)
        ).scalars().all()
        rows = []
        for v in items:
            row = VideoOut.model_validate(v)
            row.can_open = video_role(db, v, principal) is not None
            rows.append(row)
        return VideoListOut(
            items=rows,
            total=total,
            page=page,
            page_size=page_size,
        )

    rows = []
    for v in db.execute(stmt).scalars():
        if video_role(db, v, principal) is None:
            continue
        row = VideoOut.model_validate(v)
        row.can_open = True
        rows.append(row)
    total = len(rows)
    start = (page - 1) * page_size
    return VideoListOut(
        items=rows[start : start + page_size],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=VideoOut, status_code=201)
def create_video(
    payload: VideoCreate,
    request: Request,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    enforce_rate_limit(
        f"video_create:{_client_ip(request)}",
        settings.video_create_rate_limit,
        settings.video_create_rate_window,
    )
    source_url = validate_source_url(payload.source_url)
    if payload.folder_id is not None:
        folder = db.get(Folder, payload.folder_id)
        if folder is None:
            raise AppError(404, "not_found", "Folder not found.", [{"id": str(payload.folder_id)}])
        # Ingesting *into* a folder is an edit of that dataset.
        require_manage(db, folder, principal)

    # Idempotency: the same URL (that is not failed) must not be duplicated.
    # Only against what the caller can already read — see `_dedupe_scope`.
    #
    # The check below is a read followed by a write, so on its own it loses to
    # a double-tapped button: two requests both find nothing and both insert,
    # and the same media is downloaded and transcribed twice. A transaction
    # advisory lock keyed on the URL serialises them, so the second request
    # always sees the row the first one wrote and answers the usual 409.
    _lock_on_source_url(db, source_url)
    scope = _dedupe_scope(db, principal)
    stmt = select(Video).where(Video.source_url == source_url, Video.status != "failed")
    if scope is not None:
        stmt = stmt.where(scope)
    existing = db.execute(stmt).scalars().first()
    if existing is not None:
        raise AppError(
            409,
            "conflict",
            # A teammate pasting a link that is already in the dataset should
            # land on the existing unit, so the message says so and the id is
            # in `details` for the client to open it.
            "This link is already in the dataset. Open the existing unit instead of "
            "downloading it again.",
            [
                {
                    "field": "source_url",
                    "issue": "duplicate_video",
                    "video_id": str(existing.id),
                }
            ],
            # The existing resource is returned alongside the error envelope so
            # clients do not have to create a duplicate.
            extra={"video": VideoOut.model_validate(existing).model_dump(mode="json")},
        )

    # TikTok/Facebook answer an anonymous server with a bot check, so a new
    # gated download must run under the requesting user's own session: they
    # need to have consented and uploaded their cookies through the bot.
    require_session_consent(db, principal, source_url)

    position = 0
    if payload.folder_id is not None:
        position = db.execute(
            select(func.coalesce(func.max(Video.position), -1) + 1).where(
                Video.folder_id == payload.folder_id
            )
        ).scalar_one()

    video = Video(
        source_url=source_url,
        platform=detect_platform(source_url),
        title=payload.title,
        notes=payload.notes,
        folder_id=payload.folder_id,
        position=position,
        status="pending",
        progress=0,
        created_by=principal.actor,
    )
    db.add(video)
    db.flush()

    queue.enqueue(
        db,
        "download",
        video_id=video.id,
        payload={"chain": ["extract_audio", "transcribe"] if payload.start_transcription else []},
        priority=100,
    )
    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)


ALLOWED_UPLOAD_SUFFIXES = {
    ".mp4",
    ".m4v",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
}


def _safe_suffix(filename: str | None) -> str:
    """Allowed lowercase suffix from a client-supplied filename, or ''."""
    if not filename:
        return ""
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in ALLOWED_UPLOAD_SUFFIXES else ""


@router.post("/upload", response_model=VideoOut, status_code=201)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    folder_id: Optional[UUID] = Form(None),
    start_transcription: bool = Form(True),
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    """Ingest a media file the operator already has on disk.

    Ingestion must not depend on a platform serving a download: TikTok and
    YouTube refuse most server-side requests, so an operator who has the file
    can upload it directly and get the same pipeline (checksum, audio
    extraction, transcription) behind it.
    """
    enforce_rate_limit(
        f"video_create:{_client_ip(request)}",
        settings.video_create_rate_limit,
        settings.video_create_rate_window,
    )

    suffix = _safe_suffix(file.filename)
    if not suffix:
        raise AppError(
            415,
            "unsupported_media_type",
            "Unsupported file type. Upload an audio or video file.",
            [
                {
                    "field": "file",
                    "issue": "unsupported_type",
                    "filename": file.filename,
                    "allowed": sorted(ALLOWED_UPLOAD_SUFFIXES),
                }
            ],
        )

    limit_bytes = settings.max_upload_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit_bytes:
        raise AppError(
            413,
            "payload_too_large",
            f"File exceeds the {settings.max_upload_mb} MB upload limit.",
            [{"limit_bytes": limit_bytes}],
        )

    if folder_id is not None:
        folder = db.get(Folder, folder_id)
        if folder is None:
            raise AppError(404, "not_found", "Folder not found.", [{"id": str(folder_id)}])
        require_manage(db, folder, principal)

    video = Video(
        source_url=f"upload://{Path(file.filename or 'upload').name}",
        platform="direct",
        title=title or None,
        notes=notes or None,
        folder_id=folder_id,
        status="downloading",
        progress=0,
        created_by=principal.actor,
    )
    db.add(video)
    db.flush()

    dest = storage.media_file(video.id, suffix)
    digest = hashlib.sha256()
    written = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit_bytes:
                    raise AppError(
                        413,
                        "payload_too_large",
                        f"File exceeds the {settings.max_upload_mb} MB upload limit.",
                        [{"limit_bytes": limit_bytes}],
                    )
                digest.update(chunk)
                out.write(chunk)
    except AppError:
        _discard_upload(db, video, dest)
        raise
    except OSError as exc:
        _discard_upload(db, video, dest)
        raise AppError(
            500, "upload_failed", "Could not write the uploaded file to disk."
        ) from exc

    if written == 0:
        _discard_upload(db, video, dest)
        raise AppError(
            422, "validation_failed", "The uploaded file is empty.", [{"field": "file"}]
        )

    checksum = digest.hexdigest()
    # Idempotency: the same bytes are the same asset, whatever the filename.
    scope = _dedupe_scope(db, principal)
    stmt = select(Video).where(
        Video.checksum_sha256 == checksum, Video.status != "failed"
    )
    if scope is not None:
        stmt = stmt.where(scope)
    duplicate = db.execute(stmt).scalars().first()
    if duplicate is not None:
        _discard_upload(db, video, dest)
        raise AppError(
            409,
            "conflict",
            "This file has already been uploaded.",
            [{"field": "file", "issue": "duplicate_video", "video_id": str(duplicate.id)}],
            extra={"video": VideoOut.model_validate(duplicate).model_dump(mode="json")},
        )

    video.checksum_sha256 = checksum
    video.file_size = written
    video.local_path = str(dest)
    # curl and many clients send application/octet-stream for any file, which
    # tells the dashboard's <video> element nothing. Fall back to the suffix.
    declared_type = (file.content_type or "").strip()
    if not declared_type or declared_type == "application/octet-stream":
        video.mime_type = storage.guess_mime(dest)
    else:
        video.mime_type = declared_type
    video.duration_seconds = asr.probe_duration(dest)
    video.status = "downloaded"
    video.progress = 100

    db.flush()
    queue.enqueue(
        db,
        "extract_audio",
        video_id=video.id,
        payload={"chain": ["transcribe"] if start_transcription else []},
        priority=100,
    )
    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)


def _discard_upload(db: Session, video: Video, dest: Path) -> None:
    """Remove a half-created upload so a rejected request leaves nothing behind."""
    try:
        if dest.exists():
            dest.unlink()
    except OSError:
        logger.warning("could not remove rejected upload at %s", dest)
    db.rollback()


@router.get("/{video_id}", response_model=VideoDetailOut)
def get_video(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_optional_principal),
) -> VideoDetailOut:
    video = get_video_or_404(db, video_id)
    # A video is as readable as the folder it lives in (instruction.txt #3, #9).
    require_video(db, video, principal)
    transcripts = db.execute(
        select(Transcript)
        .where(Transcript.video_id == video_id)
        .order_by(Transcript.version.desc())
    ).scalars().all()
    out = VideoDetailOut.model_validate(video)
    out.transcripts = [TranscriptOut.model_validate(t) for t in transcripts]
    # Readable names for the actor strings, resolved here and nowhere near the
    # stored values: `created_by` stays exactly what it was, because access
    # control and the worker both depend on the raw string.
    out.created_by_display = display_name(db, video.created_by)
    out.collected_by_display = display_name(db, video.collected_by)
    # Job history is operational: only signed-in users see the queue internals.
    if principal.is_signed_in or principal.role == "telegram":
        jobs = db.execute(
            select(Job).where(Job.video_id == video_id).order_by(Job.created_at.desc())
        ).scalars().all()
        out.jobs = [JobOut.model_validate(j) for j in jobs]
    return out


@router.patch("/{video_id}", response_model=VideoOut)
def update_video(
    video_id: UUID,
    payload: VideoUpdate,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    video = get_video_or_404(db, video_id)
    require_video(db, video, principal, need="editor")
    data = payload.model_dump(exclude_unset=True)
    # `folder_id` is a move, not a field edit: entering a folder is a write on
    # that folder. The guard above covers the folder being left; without this an
    # editor of folder A could drop a video into a private folder B they cannot
    # see, which is the same hole as `PUT /videos/{id}/folder`.
    if data.get("folder_id") is not None:
        require_manage(db, folder_or_404(db, data["folder_id"]), principal)
    for field, value in data.items():
        setattr(video, field, value)
    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)


@router.delete("/{video_id}", status_code=204)
def delete_video(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> Response:
    video = get_video_or_404(db, video_id)
    require_video(db, video, principal, need="editor")
    db.execute(Transcript.__table__.delete().where(Transcript.video_id == video_id))
    db.execute(Merge.__table__.delete().where(Merge.video_id == video_id))
    db.execute(Job.__table__.delete().where(Job.video_id == video_id))
    db.delete(video)
    db.commit()
    storage.delete_video_files(video_id)
    return Response(status_code=204)


@router.post("/{video_id}/copy", response_model=VideoOut, status_code=201)
def copy_video(
    video_id: UUID,
    payload: VideoCopyIn,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    """Fork a video into a folder the caller manages, leaving the original alone.

    The complement of moving (`PUT /videos/{id}/folder`). A move is curation: it
    needs edit rights on the source and takes the unit out of wherever it was,
    so a public card stops being public. A copy is perusal: it needs only *read*
    on the source — the caller has already seen it — and writes into a folder
    they manage. Without it, a collector who wants their own annotated copy of a
    colleague's public unit would have to re-download and re-transcribe it.

    The copy is a snapshot, not a shared pointer. `tvna_serial` stays NULL (it is
    unique, and it names the item in the world, not the row); the Telegram mirror
    ids are dropped (they point at one chat message); the transcript is cloned as
    version 1 so later edits cannot touch the original; and the media bytes are
    hard-linked, so a fork costs no disk, no download and no ASR run.
    """
    source = get_video_or_404(db, video_id)
    require_video(db, source, principal, need="viewer")

    folder = folder_or_404(db, payload.folder_id)
    require_manage(db, folder, principal)
    if source.folder_id == folder.id:
        raise AppError(
            422,
            "validation_failed",
            "That video is already in this folder.",
            [{"field": "folder_id", "issue": "same_folder"}],
        )

    # A copy enqueues no job, so copying a row whose bytes never arrived would
    # create a card stuck at "processing" forever.
    media = storage.find_media_file(source.id)
    if media is None:
        raise AppError(
            409,
            "not_ready",
            "This video has no media on disk yet, so there is nothing to copy.",
            [{"video_id": str(source.id), "status": source.status}],
        )

    position = payload.position
    if position is None:
        position = db.execute(
            select(func.coalesce(func.max(Video.position), -1) + 1).where(
                Video.folder_id == folder.id
            )
        ).scalar_one()

    transcript = zip_export.current_transcript(db, source.id)
    fork = Video(
        source_url=source.source_url,
        platform=source.platform,
        title=source.title,
        notes=source.notes,
        folder_id=folder.id,
        position=position,
        copied_from_id=source.id,
        status="transcribed" if transcript is not None else "downloaded",
        progress=100,
        duration_seconds=source.duration_seconds,
        mime_type=source.mime_type,
        thumbnail_url=source.thumbnail_url,
        # The TvNa collection fields describe the item, so they travel with the
        # copy — all but the serial, which is unique and names the original.
        source_type=source.source_type,
        author_handle=source.author_handle,
        date_posted=source.date_posted,
        caption=source.caption,
        location_tag=source.location_tag,
        collection_method=source.collection_method,
        collected_by=source.collected_by,
        collection_date=source.collection_date,
        created_by=principal.actor,
    )
    db.add(fork)
    db.flush()

    storage.clone_video_files(source.id, fork.id)
    fork.local_path = str(storage.find_media_file(fork.id) or media)

    if transcript is not None:
        db.add(
            Transcript(
                video_id=fork.id,
                version=1,
                source=transcript.source,
                language=transcript.language,
                content=transcript.content,
                segments=copy.deepcopy(transcript.segments),
                status=transcript.status,
                cleaned=transcript.cleaned,
                created_by=principal.actor,
            )
        )

    db.commit()
    db.refresh(fork)
    return VideoOut.model_validate(fork)


@router.post("/{video_id}/retry", response_model=JobOut, status_code=202)
def retry_video(
    video_id: UUID,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> JobOut:
    video = get_video_or_404(db, video_id)
    require_video(db, video, principal, need="editor")

    # A double-tapped Retry must not queue a second download of the same file:
    # two jobs would race over one output path and download the media twice.
    pending = db.execute(
        select(Job)
        .where(
            Job.video_id == video.id,
            Job.kind == "download",
            Job.status.in_(("queued", "running")),
        )
        .order_by(Job.created_at.desc())
    ).scalars().first()
    if pending is not None:
        raise AppError(
            409,
            "job_in_progress",
            "A download for this video is already queued or running.",
            [{"job_id": str(pending.id), "status": pending.status}],
            extra={"job": JobOut.model_validate(pending).model_dump(mode="json")},
        )

    job = queue.enqueue(
        db,
        "download",
        video_id=video.id,
        payload={"chain": ["extract_audio", "transcribe"]},
        priority=50,
    )
    if video.status in ("failed", "pending", "quarantined"):
        video.status = "pending"
        video.error = None
        video.progress = 0
    db.commit()
    db.refresh(job)
    return JobOut.model_validate(job)


# ------------------------------------------------------------------ media


def _parse_range(header: str, size: int) -> tuple[int, int]:
    """Parse a single-range `Range: bytes=...` header. Raises ValueError when invalid."""
    spec = header.strip()
    if not spec.lower().startswith("bytes="):
        raise ValueError("unsupported range unit")
    spec = spec.split("=", 1)[1].strip()
    if "," in spec:
        raise ValueError("multiple ranges are not supported")
    if "-" not in spec:
        raise ValueError("malformed range")
    start_raw, end_raw = spec.split("-", 1)
    start_raw, end_raw = start_raw.strip(), end_raw.strip()
    if start_raw == "":
        # suffix range: last N bytes
        if end_raw == "":
            raise ValueError("malformed range")
        length = int(end_raw)
        if length <= 0:
            raise ValueError("malformed range")
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else size - 1
    if start > end or start >= size:
        raise ValueError("range not satisfiable")
    end = min(end, size - 1)
    return start, end


def _authorize_media(db: Session, video: Video, principal) -> None:
    """Gate media on the folder it lives in, not just on knowing the UUID.

    The public portal serves cards from folders, so a video's media is as
    public as its folder. An unfiled video has no folder, so it belongs to
    whoever ingested it — `video_role` applies the creator rule that
    `GET /videos/{id}` already uses. Letting any signed-in member through here
    made media and audio wider than the metadata for the same resource.
    """
    if video.folder_id is None:
        if video_role(db, video, principal) is None:
            raise AppError(404, "not_found", "Media is not available yet for this video.")
        return
    folder = db.get(Folder, video.folder_id)
    if folder is not None:
        require_read(db, folder, principal)


def _open_cloud_stream(
    db: Session, video: Video, range_header: Optional[str]
) -> Response:
    """Stream a cloud-stored video from the contributor's drive.

    The API proxies (the browser can't hold the user's drive token), refreshing
    the grant from the stored link. A dead link surfaces as the same
    `cloud_file_unavailable` the cards warn about, not a bare 500.
    """
    link = db.get(DriveLink, video.drive_link_id) if video.drive_link_id else None
    if link is None or not video.drive_file_id:
        raise AppError(404, "not_found", "Media is not available yet for this video.")
    try:
        access = cloud_providers.refresh_access_token(video.storage_backend, decrypt_token(link.refresh_token))
    except AppError as exc:
        # A dead/revoked grant -> storage unavailable, not an auth error
        raise AppError(
            404,
            "cloud_file_unavailable",
            "The file is not available — reconnect your drive to restore access.",
        ) from exc
    try:
        read = cloud_providers.open_download(
            video.storage_backend, access, video.drive_file_id, range_header
        )
    except AppError:
        raise

    headers: dict[str, str] = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'inline; filename="{video.drive_file_id}"',
    }
    # Forward Content-Range from provider (present on 206 responses)
    if "Content-Range" in read.headers:
        headers["Content-Range"] = read.headers["Content-Range"]
    # Compute correct Content-Length
    if read.status == 206 and "Content-Range" in read.headers:
        # Parse partial length from Content-Range: bytes start-end/total
        try:
            range_part = read.headers["Content-Range"].split(" ", 1)[1]
            byte_range = range_part.split("/")[0]
            start, end = byte_range.split("-")
            headers["Content-Length"] = str(int(end) - int(start) + 1)
        except Exception:  # noqa: BLE001
            if read.size:
                headers["Content-Length"] = str(read.size)
    elif read.size:
        headers["Content-Length"] = str(read.size)

    return StreamingResponse(
        read.body,
        status_code=read.status,
        media_type=read.headers.get("Content-Type", video.mime_type or "application/octet-stream"),
        headers=headers,
    )


@router.get("/{video_id}/media")
def stream_media(
    video_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
    principal=Depends(get_optional_principal),
) -> Response:
    video = get_video_or_404(db, video_id)
    _authorize_media(db, video, principal)
    range_header = request.headers.get("range")
    if video.storage_backend:
        return _open_cloud_stream(db, video, range_header)
    path = storage.find_media_file(video_id)
    if path is None or not path.exists():
        raise AppError(404, "not_found", "Media is not available yet for this video.")
    size = path.stat().st_size
    mime = video.mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    if not range_header:
        return StreamingResponse(
            storage.iter_file(path, 0, size - 1),
            media_type=mime,
            headers={
                "Content-Length": str(size),
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'inline; filename="{path.name}"',
            },
        )

    try:
        start, end = _parse_range(range_header, size)
    except ValueError:
        return JSONResponse(
            status_code=416,
            content={
                "error": {
                    "code": "range_not_satisfiable",
                    "message": "The requested range is not satisfiable.",
                    "details": [{"size": size, "range": range_header}],
                }
            },
            headers={"Content-Range": f"bytes */{size}"},
        )

    return StreamingResponse(
        storage.iter_file(path, start, end),
        status_code=206,
        media_type=mime,
        headers={
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{path.name}"',
        },
    )


@router.get("/{video_id}/audio")
def download_audio(
    video_id: UUID,
    format: str = Query("wav"),
    db: Session = Depends(get_db),
    principal=Depends(get_optional_principal),
) -> Response:
    """The extracted audio, for download from a public data card.

    `format=wav` (the default) is the 16 kHz mono ASR artifact the data cards
    link to; `format=mp3` is the higher-quality listening copy.
    """
    video = get_video_or_404(db, video_id)
    _authorize_media(db, video, principal)
    if format not in ("wav", "mp3"):
        raise AppError(422, "validation_failed", "format must be 'wav' or 'mp3'.")
    if format == "mp3":
        path = storage.audio_mp3_file(video_id)
        if not path.exists():
            raise AppError(404, "not_found", "MP3 has not been extracted for this video.")
        return StreamingResponse(
            storage.iter_file(path),
            media_type="audio/mpeg",
            headers={
                "Content-Length": str(path.stat().st_size),
                "Content-Disposition": f'attachment; filename="{video_id}-audio.mp3"',
            },
        )
    path = storage.audio_file(video_id)
    if not path.exists():
        raise AppError(404, "not_found", "Audio has not been extracted for this video.")
    return StreamingResponse(
        storage.iter_file(path),
        media_type="audio/wav",
        headers={
            "Content-Length": str(path.stat().st_size),
            "Content-Disposition": f'attachment; filename="{video_id}-audio.wav"',
        },
    )


# ------------------------------------------------------------------ TvNa record


def _allocate_tvna_serial(db: Session) -> int:
    """The next PP-NNN number, allocated under a lock.

    Two collectors recording at the same moment must not both get PP-007. The
    advisory lock serialises the read-then-write on Postgres; the unique index
    on `tvna_serial` is the backstop if a second process ever slips through.
    """
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        db.execute(text("select pg_advisory_xact_lock(hashtext('recsterio.tvna_serial'))"))
    highest = db.execute(select(func.max(Video.tvna_serial))).scalar_one()
    return int(highest or 0) + 1


@router.post("/{video_id}/tvna", response_model=VideoOut)
def record_tvna(
    video_id: UUID,
    payload: TvnaRecordIn,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    """Attach the TvNa collection record (instruction.txt #6).

    Idempotent in the sense that matters to a collector: re-recording the same
    video updates the fields but keeps the PP-NNN number it was already given,
    because that number is what appears in the spreadsheet and the report.
    """
    video = get_video_or_404(db, video_id)
    require_video(db, video, principal, need="editor")

    if video.tvna_serial is None:
        video.tvna_serial = _allocate_tvna_serial(db)

    video.location_tag = payload.location_tag.strip()
    video.source_type = payload.source_type
    video.collection_method = payload.collection_method
    # Auto-detected values are kept when the collector leaves the field blank:
    # the downloader usually knows the author and caption better than a human
    # retyping them does.
    if payload.author_handle:
        video.author_handle = payload.author_handle.strip()
    if payload.date_posted:
        video.date_posted = payload.date_posted.strip()
    if payload.caption:
        video.caption = payload.caption.strip()
    video.collected_by = (payload.collected_by or principal.actor).strip()
    video.collection_date = (payload.collection_date or _today()).strip()
    if payload.notes is not None:
        video.notes = payload.notes.strip() or None

    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)


def _today() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()


@router.post("/{video_id}/telegram", response_model=VideoOut)
def link_telegram(
    video_id: UUID,
    payload: TelegramLinkIn,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> VideoOut:
    """Record the Telegram copy of this video (instruction.txt #4).

    The bot calls this after it has uploaded the file to a chat, so the
    catalogue knows where a re-fetchable copy lives. Only files under the
    Bot API's 20 MB `getFile` ceiling can actually be pulled back — the column
    records the reference either way.
    """
    video = get_video_or_404(db, video_id)
    require_video(db, video, principal, need="editor")
    # The pair must describe one message: a re-link with a new file_id but no
    # message_id clears the old one instead of leaving it pointing at a
    # message that no longer holds this file.
    video.telegram_file_id = payload.file_id.strip()
    video.telegram_message_id = payload.message_id
    db.commit()
    db.refresh(video)
    return VideoOut.model_validate(video)
