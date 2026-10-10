"""Stage 6 — reranking ~50 candidates down to the top 5.

A cross-encoder reads the question against each candidate together,
rather than comparing two vectors produced independently, which is why
it is more accurate than retrieval and too slow to run over a whole
corpus. Hence the pipeline's shape: retrieve ~50 cheaply, rerank those.

Phase 2 exists to measure the gain rather than assume it. Phase 1
measured recall@5 at 0.750 against ceiling while recall@50 was 0.938 —
the right passage is nearly always retrieved but often not ranked high
enough to reach the generator. Closing that gap is this stage's whole
job, and `rerank_delta` below is what proves it did.

Every scored candidate carries the reranker's name. The abstention
floor in app.query.score_gate is calibrated against one specific score
scale and refuses to compare against another, so this is load-bearing,
not bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.query.retrieve import Scored


class RerankerLike(Protocol):
    name: str

    def rerank_sync(
        self, question: str, texts: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]: ...


@dataclass(frozen=True)
class RerankedHit:
    chunk_id: str
    document: str
    text: str
    score: float
    reranker: str
    page: int | None = None
    section: str | None = None
    has_warning: bool = False
    warnings: tuple[str, ...] = ()
    retrieval_rank: int | None = None
    reranked_rank: int | None = None


def rerank(
    question: str,
    candidates: list[Scored],
    reranker: RerankerLike,
    top_k: int = 5,
) -> list[RerankedHit]:
    if not candidates:
        return []

    order = reranker.rerank_sync(
        question, [c.chunk.text for c in candidates], top_k=top_k
    )
    return [
        RerankedHit(
            chunk_id=candidates[idx].chunk.chunk_id,
            document=candidates[idx].chunk.document,
            text=candidates[idx].chunk.text,
            score=score,
            reranker=reranker.name,
            page=candidates[idx].chunk.page,
            section=candidates[idx].chunk.section,
            has_warning=candidates[idx].chunk.has_warning,
            warnings=candidates[idx].chunk.warnings,
            retrieval_rank=idx + 1,
            reranked_rank=new_rank,
        )
        for new_rank, (idx, score) in enumerate(order, start=1)
    ]
