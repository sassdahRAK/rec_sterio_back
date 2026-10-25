"""Cross-process worker heartbeat.

The health endpoint must report ``worker_alive=true`` even when the worker
runs as a *separate* process (``python -m app.worker``) from the API server.
An in-process flag cannot cross a process boundary, so the worker loop also
refreshes a small JSON file that :func:`app.worker.main.worker_alive` reads.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger("recsterio.worker.heartbeat")


def heartbeat_path() -> Path:
    p = Path(settings.worker_heartbeat_file).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def write_heartbeat(pid: Optional[int] = None) -> None:
    """Atomically refresh the heartbeat file (best effort; never raises)."""
    try:
        p = heartbeat_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        # Unique per process *and* thread so concurrent workers never race on
        # the same temp file.
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(
            json.dumps({"pid": pid or os.getpid(), "ts": time.time()}),
            encoding="utf-8",
        )
        tmp.replace(p)
    except OSError:  # pragma: no cover - a missing heartbeat must not kill the worker
        logger.exception("failed to write worker heartbeat")


def heartbeat_age() -> Optional[float]:
    """Seconds since the last heartbeat, or None when unavailable/stale-unreadable."""
    try:
        data = json.loads(heartbeat_path().read_text(encoding="utf-8"))
        return max(0.0, time.time() - float(data.get("ts", 0)))
    except (OSError, ValueError, TypeError):
        return None
