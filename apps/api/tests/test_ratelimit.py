"""Per-organization token-bucket rate limiting.

Disabled for the rest of the suite (see conftest), so this module turns it back on and drives
the limiter directly.
"""

from __future__ import annotations

import time

import pytest

from aperture.core import ratelimit


@pytest.fixture(autouse=True)
def _clean_buckets():
    ratelimit.reset()
    yield
    ratelimit.reset()


def test_a_burst_up_to_capacity_is_allowed():
    """The property a fixed window cannot give: spend a day's batch in one go."""
    for _ in range(10):
        assert ratelimit._consume("k", capacity=10, refill_per_second=1) == 0.0


def test_exceeding_capacity_reports_a_wait():
    for _ in range(10):
        ratelimit._consume("k", capacity=10, refill_per_second=1)
    wait = ratelimit._consume("k", capacity=10, refill_per_second=1)
    assert wait > 0


def test_tokens_refill_over_time():
    for _ in range(5):
        ratelimit._consume("k", capacity=5, refill_per_second=1000)
    assert ratelimit._consume("k", capacity=5, refill_per_second=1000) > 0
    time.sleep(0.01)  # 1000 tokens/s -> ~10 tokens back
    assert ratelimit._consume("k", capacity=5, refill_per_second=1000) == 0.0


def test_refill_never_exceeds_capacity():
    """An idle tenant must not bank unlimited credit and then flood."""
    ratelimit._consume("k", capacity=3, refill_per_second=1000)
    time.sleep(0.02)  # would be ~20 tokens if uncapped
    allowed = sum(1 for _ in range(10) if ratelimit._consume("k", capacity=3, refill_per_second=1000) == 0.0)
    assert allowed == 3


def test_buckets_are_isolated_per_key():
    """One tenant exhausting its budget must not affect another."""
    for _ in range(5):
        ratelimit._consume("org-a", capacity=5, refill_per_second=0.001)
    assert ratelimit._consume("org-a", capacity=5, refill_per_second=0.001) > 0
    assert ratelimit._consume("org-b", capacity=5, refill_per_second=0.001) == 0.0


def test_endpoint_budgets_are_separate():
    """A flood of uploads must not lock the caller out of reading the dashboard."""
    for _ in range(5):
        ratelimit._consume("ingest:org", capacity=5, refill_per_second=0.001)
    assert ratelimit._consume("ingest:org", capacity=5, refill_per_second=0.001) > 0
    assert ratelimit._consume("recompute:org", capacity=5, refill_per_second=0.001) == 0.0


# --- through the API --------------------------------------------------------------------------


@pytest.fixture
def limited(monkeypatch):
    """Enable the limiter and shrink the recompute budget so the test is fast."""
    from aperture.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(ratelimit.recompute_limit, "capacity", 2)
    monkeypatch.setattr(ratelimit.recompute_limit, "refill_per_second", 0.0001)
    yield


def test_recompute_returns_429_once_the_budget_is_spent(client, auth, limited):
    codes = [client.post("/v1/clusters/recompute", headers=auth).status_code for _ in range(4)]
    assert codes[:2] == [202, 202], "recompute queues the work and answers 202"
    assert 429 in codes[2:], f"expected a 429 after the budget was spent, got {codes}"


def test_429_tells_the_caller_when_to_retry(client, auth, limited):
    for _ in range(3):
        response = client.post("/v1/clusters/recompute", headers=auth)
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert int(response.headers["Retry-After"]) >= 1


def test_limits_are_scoped_per_org(client, limited):
    """other-org must still be served after acme has spent its budget."""
    for _ in range(3):
        client.post("/v1/clusters/recompute", headers={"X-API-Key": "demo-key"})
    assert client.post("/v1/clusters/recompute", headers={"X-API-Key": "demo-key"}).status_code == 429
    assert client.post("/v1/clusters/recompute", headers={"X-API-Key": "other-key"}).status_code == 202


def test_limiter_is_inert_when_disabled(client, auth):
    """Default test configuration: the suite must not trip over budgets."""
    codes = {client.post("/v1/clusters/recompute", headers=auth).status_code for _ in range(8)}
    assert codes == {202}
