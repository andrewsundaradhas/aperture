"""The worker loop.

Two ways to run it, because two situations need different things:

* **A separate process** — `python -m aperture.jobs.worker`. What production uses: the worker
  scales and restarts independently of the API, and a job that exhausts memory takes down the
  worker rather than the thing serving requests.
* **In-process, on a daemon thread** — `APERTURE_INLINE_WORKER=true`, the default. Without it
  `uvicorn aperture.main:app` alone would leave every queued job unclaimed, and the quick-start
  in the README would quietly do nothing. Convenience for a single-machine run, not a
  production topology; `docker-compose.yml` sets it false and runs the real worker.

The loop backs off when the queue is empty so an idle deployment is not a spin loop, and keeps
going after a handler raises — `run_once` has already recorded the failure on the row.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from aperture.core.db import SessionLocal
from aperture.jobs.queue import run_once

logger = logging.getLogger(__name__)

IDLE_SLEEP_SECONDS = 1.0
BUSY_SLEEP_SECONDS = 0.0


def drain(kinds: list[str] | None = None, max_jobs: int | None = None) -> int:
    """Run queued jobs until the queue is empty. Returns how many ran.

    The synchronous entry point tests use, and what the inline worker calls each tick.
    """
    ran = 0
    db = SessionLocal()
    try:
        while max_jobs is None or ran < max_jobs:
            if run_once(db, kinds) is None:
                break
            ran += 1
    finally:
        db.close()
    return ran


def run_forever(kinds: list[str] | None = None, stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    logger.info("worker started (kinds=%s)", kinds or "all")
    while not stop.is_set():
        try:
            worked = drain(kinds, max_jobs=1)
        except Exception:  # noqa: BLE001 - a database blip must not end the worker
            logger.exception("worker loop error; continuing")
            worked = 0
        stop.wait(IDLE_SLEEP_SECONDS if not worked else BUSY_SLEEP_SECONDS)
    logger.info("worker stopped")


_inline_stop: threading.Event | None = None
_inline_thread: threading.Thread | None = None


def start_inline_worker() -> None:
    """Start the in-process worker thread. Idempotent."""
    global _inline_stop, _inline_thread
    if _inline_thread is not None and _inline_thread.is_alive():
        return
    _inline_stop = threading.Event()
    _inline_thread = threading.Thread(
        target=run_forever, kwargs={"stop": _inline_stop}, name="aperture-inline-worker", daemon=True
    )
    _inline_thread.start()
    logger.info("inline worker thread started")


def stop_inline_worker(timeout: float = 5.0) -> None:
    if _inline_stop is not None:
        _inline_stop.set()
    if _inline_thread is not None:
        _inline_thread.join(timeout=timeout)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    stop = threading.Event()
    # Answer SIGTERM so `docker compose down` is a clean stop, not a kill: the in-flight job
    # finishes and the next claim never happens.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    run_forever(stop=stop)


if __name__ == "__main__":
    main()
