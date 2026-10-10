"""POST /query — synchronous query endpoint.

Accepts a question (and optional session id for follow-up rewriting),
runs it through app.query.pipeline, and returns an answer with citations
or an abstention with the relevant sections. Latency-budgeted per the
architecture's ~3s target.

The response shape puts `passages` on every outcome, not only on
answered ones. A refusal that returns the closest sections is the
product working, so the client renders sources whether or not there is
an answer — the architecture's rule is that the system never returns
nothing.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    documents: list[str] | None = None


class Citation(BaseModel):
    document: str
    page: int | None
    section: str | None
    text: str
    score: float
    has_warning: bool


class QueryResponse(BaseModel):
    outcome: str
    answered: bool
    answer: str | None
    reason: str
    passages: list[Citation]
    timings_ms: dict[str, float]


@router.post("/query", response_model=QueryResponse)
async def post_query(body: QueryRequest, request: Request) -> QueryResponse:
    result = request.app.state.pipeline.run(
        body.question, documents=set(body.documents) if body.documents else None
    )
    return QueryResponse(
        outcome=str(result.outcome),
        answered=result.answered,
        answer=result.answer,
        reason=result.reason,
        passages=[
            Citation(
                document=p.document,
                page=p.page,
                section=p.section,
                text=p.text,
                score=round(p.score, 4),
                has_warning=p.has_warning,
            )
            for p in result.passages
        ],
        timings_ms=result.timings_ms,
    )
