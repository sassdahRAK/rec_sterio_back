"""Worker entrypoint: `python -m app.worker.main`.

The worker is also started in-process by the FastAPI app (settings.worker_autostart).
"""
from __future__ import annotations

import logging
import signal
import threading
import time

from app.config import settings
from app.db import SessionLocal, init_db
from app.worker import heartbeat, queue
from app.worker.runner import run_job

logger = logging.getLogger("recsterio.worker")

_stop = threading.Event()
_last_heartbeat = 0.0


def worker_alive(max_age_seconds: float = 60.0) -> bool:
    """True when a worker has checked in recently.

    Works both for the in-process worker started by the app and for a worker
    running as a separate ``python -m app.worker`` process (via the heartbeat
    file).
    """
    if _last_heartbeat and (time.time() - _last_heartbeat) <= max_age_seconds:
        return True
    age = heartbeat.heartbeat_age()
    return age is not None and age <= max_age_seconds


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):  # noqa: ANN001
        logger.info("worker received signal %s, shutting down", signum)
        _stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # pragma: no cover - non main thread
            pass


def worker_loop(thread_name: str = "worker") -> None:
    global _last_heartbeat
    logger.info("worker %s started (poll=%.1fs)", thread_name, settings.worker_poll_seconds)
    while not _stop.is_set():
        try:
            _last_heartbeat = time.time()
            heartbeat.write_heartbeat()
            with SessionLocal() as db:
                job = queue.claim_job(db)
                if job is None:
                    _stop.wait(settings.worker_poll_seconds)
                    continue
                logger.info("claimed job %s kind=%s video=%s", job.id, job.kind, job.video_id)
                run_job(db, job)
        except Exception:  # noqa: BLE001 - the worker must never die
            logger.exception("worker loop error")
            _stop.wait(settings.worker_poll_seconds)
    logger.info("worker %s stopped", thread_name)


def start_worker_threads(count: int | None = None) -> list[threading.Thread]:
    count = count if count is not None else max(1, settings.worker_concurrency)
    threads: list[threading.Thread] = []
    for i in range(count):
        t = threading.Thread(target=worker_loop, args=(f"worker-{i}",), daemon=True)
        t.start()
        threads.append(t)
    return threads


def main() -> None:  # pragma: no cover - process entrypoint
    logging.basicConfig(level=settings.log_level)
    init_db()
    _install_signal_handlers()
    start_worker_threads()
    while not _stop.is_set():
        time.sleep(0.5)


if __name__ == "__main__":  # pragma: no cover
    main()
