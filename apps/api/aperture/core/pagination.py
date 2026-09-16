"""Shared pagination for list endpoints.

One style across the whole API — `limit` + `offset` — rather than cursors. Cursors scale better
past very deep offsets, but they cannot express "jump to page 7", which the dashboard's episode
table needs, and every list here is org-scoped and indexed so deep offsets stay cheap.

Unbounded lists were the real problem: `GET /v1/episodes` returned every episode an org had
ever uploaded, so the response grew without limit as a fleet ran. `DEFAULT_LIMIT` bounds an
un-parameterised call, and `MAX_LIMIT` bounds a hostile one.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Query

DEFAULT_LIMIT = 50
MAX_LIMIT = 500


@dataclass
class Page:
    limit: int
    offset: int


def page_params(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Rows to return."),
    offset: int = Query(0, ge=0, description="Rows to skip."),
) -> Page:
    """FastAPI dependency giving every list endpoint the same two knobs."""
    return Page(limit=limit, offset=offset)
