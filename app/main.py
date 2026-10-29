"""recsterio backend - FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import SessionLocal, init_db
from app.errors import register_exception_handlers
from app.routers import (
    auth,
    bot_users,
    cloud,
    folders,
    health,
    jobs,
    merges,
    passcodes,
    public,
    shares,
    stats,
    transcripts,
    users,
    videos,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("recsterio")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Seed the bootstrap admin user (idempotent).
    try:
        with SessionLocal() as db:
            from app.auth import seed_admin_user

            seed_admin_user(db)
    except Exception:  # noqa: BLE001
        logger.exception("admin bootstrap failed")
    if settings.worker_autostart:
        from app.worker.main import start_worker_threads

        app.state.worker_threads = start_worker_threads()
    yield
    for thread in getattr(app.state, "worker_threads", []) or []:
        thread.join(timeout=2)


app = FastAPI(
    title="recsterio API",
    version="1.0.0",
    description="Download, transcribe and merge video transcripts (Khmer/English).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # Auth is a bearer header, never a cookie (both frontends send
    # `Authorization`), so credentialed CORS is dead surface area: with it
    # set, `*` origins are the only thing FastAPI refuses, and a pinned
    # origin list invites cookie-bearing cross-site calls there is no use
    # case for. Flipping it costs the browsers nothing.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Retry-After"],
)

register_exception_handlers(app)

API_PREFIX = "/api/v1"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(stats.router, prefix=API_PREFIX)
app.include_router(videos.router, prefix=API_PREFIX)
app.include_router(transcripts.router, prefix=API_PREFIX)
app.include_router(merges.router, prefix=API_PREFIX)
app.include_router(jobs.router, prefix=API_PREFIX)
app.include_router(folders.router, prefix=API_PREFIX)
app.include_router(users.router, prefix=API_PREFIX)
app.include_router(shares.router, prefix=API_PREFIX)
app.include_router(passcodes.router, prefix=API_PREFIX)
app.include_router(public.router, prefix=API_PREFIX)
app.include_router(bot_users.router, prefix=API_PREFIX)
app.include_router(cloud.router, prefix=API_PREFIX)


@app.get("/")
def root() -> dict:
    return {"service": "recsterio", "api": API_PREFIX, "docs": "/docs"}
