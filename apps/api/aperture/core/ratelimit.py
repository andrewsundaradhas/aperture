"""Per-organization rate limiting.

**Token bucket, not fixed window.** Ingestion is bursty by nature: a fleet uploads a day of
episodes in one batch, then goes quiet. A fixed window either rejects that burst or has to be
set so high it stops limiting anything, and it also allows double the intended rate across a
window boundary. A token bucket lets a caller spend accumulated capacity in one go and then
refill steadily, which is the shape the traffic actually has.

**Scope is the organization, not the IP.** The limit exists to stop one tenant starving the
others; a tenant behind one NAT and a tenant across fifty machines should get the same budget.

**In-process storage, deliberately, for now.** Buckets live in this process's memory, so N
replicas grant N times the limit. That is honest for a single free-tier dyno and wrong for a
horizontally scaled deployment — `_BUCKETS` is the one thing to move to Redis when there is
more than one instance. The limiter never blocks or queues; it answers immediately and the
caller returns 429.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status

from aperture.core.auth import resolve_org
from aperture.core.config import get_settings
from aperture.core.models import Organization


@dataclass
class _Bucket:
    tokens: float
    updated: float = field(default_factory=time.monotonic)


_BUCKETS: dict[str, _Bucket] = {}
_LOCK = threading.Lock()


def _consume(key: str, capacity: float, refill_per_second: float, cost: float = 1.0) -> float:
    """Spend `cost` tokens. Returns seconds to wait, or 0.0 when the request may proceed."""
    now = time.monotonic()
    with _LOCK:
        bucket = _BUCKETS.get(key)
        if bucket is None:
            bucket = _BUCKETS[key] = _Bucket(tokens=capacity)
        # Refill for the elapsed time, capped at the bucket's capacity.
        bucket.tokens = min(capacity, bucket.tokens + (now - bucket.updated) * refill_per_second)
        bucket.updated = now
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return 0.0
        return (cost - bucket.tokens) / refill_per_second


def reset() -> None:
    """Test hook: forget every bucket."""
    with _LOCK:
        _BUCKETS.clear()


class RateLimit:
    """A dependency enforcing one named budget. Use a distinct `name` per endpoint class so a
    flood of uploads cannot exhaust the budget for reading the dashboard."""

    def __init__(self, name: str, capacity: int, refill_per_minute: float) -> None:
        self.name = name
        self.capacity = capacity
        self.refill_per_second = refill_per_minute / 60.0

    def __call__(
        self,
        request: Request,
        org: Organization = Depends(resolve_org),
    ) -> Organization:
        if not get_settings().rate_limit_enabled:
            return org
        wait = _consume(f"{self.name}:{org.id}", self.capacity, self.refill_per_second)
        if wait > 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit for '{self.name}' exceeded. Retry in {wait:.1f}s. "
                    f"Budget: {self.capacity} burst, {self.refill_per_second * 60:.0f}/min sustained."
                ),
                headers={"Retry-After": str(max(1, int(wait + 0.999)))},
            )
        return org


# Ingestion is the expensive path (decode, blob writes), so it gets the tighter sustained rate
# with a burst allowance big enough for a day's batch upload.
ingest_limit = RateLimit("ingest", capacity=120, refill_per_minute=60)
# Recompute is O(all failed episodes) and there is no point running it back to back.
recompute_limit = RateLimit("recompute", capacity=5, refill_per_minute=5)
