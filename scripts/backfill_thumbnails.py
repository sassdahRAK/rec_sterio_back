"""Cache a local frame for every video that has none yet.

Platform CDN thumbnail URLs are signed and expire (TikTok's do within hours),
so rows ingested before the local cache existed show a blank card. This walks
the videos whose media directory holds no `thumbnail.*` and fills it:

1. fetch the stored `thumbnail_url` (works when it is unsigned or still live);
2. fall back to re-extracting a fresh signed URL from `source_url` with yt-dlp.

Idempotent: a row that already has a cached frame is skipped. Run it with
`python -m scripts.backfill_thumbnails [--dry-run] [--limit N]`.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Video
from app.services import storage
from app.services.thumbnail import cache_from_url, derive_for_video

TITLE_WIDTH = 40


def _label(video: Video) -> str:
    title = " ".join((video.title or "").split())
    if len(title) > TITLE_WIDTH:
        title = title[:TITLE_WIDTH]
    return title or "(untitled)"


def _missing_videos(db, limit: Optional[int]) -> list[Video]:
    videos = list(
        db.execute(select(Video).order_by(Video.created_at.asc())).scalars()
    )
    missing = [v for v in videos if storage.thumbnail_file(v.id) is None]
    return missing if limit is None else missing[:limit]


def _process(video: Video) -> str:
    if video.thumbnail_url and cache_from_url(video.id, video.thumbnail_url):
        return "cached"
    derived = derive_for_video(video.source_url)
    if derived is None:
        return "failed"
    data, content_type = derived
    try:
        storage.save_thumbnail(video.id, data, content_type)
    except Exception:
        return "failed"
    return "derived"


def backfill(*, dry_run: bool = False, limit: Optional[int] = None) -> int:
    counts = {"cached": 0, "derived": 0, "failed": 0, "would-cache": 0, "would-derive": 0}
    with SessionLocal() as db:
        for video in _missing_videos(db, limit):
            if dry_run:
                result = "would-cache" if video.thumbnail_url else "would-derive"
            else:
                result = _process(video)
            counts[result] += 1
            print(f"{video.id}  {_label(video):<{TITLE_WIDTH}}  {result}")

    written = counts["cached"] + counts["derived"]
    print(
        f"cached={counts['cached']} derived={counts['derived']} "
        f"failed={counts['failed']} total_written={written}"
    )
    return 0 if counts["failed"] == 0 else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill locally cached thumbnails.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what each row would do without fetching or writing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N missing rows",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return backfill(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
