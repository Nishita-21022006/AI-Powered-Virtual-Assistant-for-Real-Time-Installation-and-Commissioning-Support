"""GET /health — liveness/readiness checks for the API process.

Checks its own dependencies (index loaded, models resolved) without
touching the ingestion worker pool, which is a separate process and has
its own health surface.

Reports which models are actually live rather than which are
configured, so a misread .env shows up here instead of as a confusing
answer later.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    stats = getattr(request.app.state, "stats", None)
    if stats is None:
        return {"status": "starting"}
    return {
        "status": "ok",
        "degraded": stats.get("generator") is None,
        **stats,
    }
