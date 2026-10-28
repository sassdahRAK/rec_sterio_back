"""Stats + health routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.access import require_operator
from app.auth import Principal, get_principal
from app.db import get_db
from app.models import Job, Merge, Transcript, Video
from app.schemas import StatsOut

router = APIRouter(tags=["stats"])


def _counts(db: Session, column, model) -> dict[str, int]:
    rows = db.execute(select(column, func.count()).group_by(column)).all()
    return {str(key): int(value) for key, value in rows}


@router.get("/stats", response_model=StatsOut)
def get_stats(
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> StatsOut:
    # Global aggregates describe the whole deployment, not one folder, so they
    # are operator data like the job queue.
    require_operator(principal)
    videos_total = db.execute(select(func.count()).select_from(Video)).scalar_one()
    transcripts_total = db.execute(select(func.count()).select_from(Transcript)).scalar_one()
    merges_total = db.execute(select(func.count()).select_from(Merge)).scalar_one()
    jobs_queued = db.execute(
        select(func.count()).select_from(Job).where(Job.status == "queued")
    ).scalar_one()
    jobs_failed = db.execute(
        select(func.count()).select_from(Job).where(Job.status.in_(("failed", "dead")))
    ).scalar_one()
    total_media_bytes = (
        db.execute(select(func.coalesce(func.sum(Video.file_size), 0))).scalar_one() or 0
    )
    return StatsOut(
        videos_total=int(videos_total),
        videos_by_status=_counts(db, Video.status, Video),
        transcripts_total=int(transcripts_total),
        transcripts_by_source=_counts(db, Transcript.source, Transcript),
        merges_total=int(merges_total),
        jobs_queued=int(jobs_queued),
        jobs_failed=int(jobs_failed),
        total_media_bytes=int(total_media_bytes),
    )
