"""Reranker protocol.

Scores ~50 candidates against the question and returns the top 5 with
their scores. Those scores are what the abstention gate thresholds on,
so the scale must be stable across model swaps or the floor has to be
recalibrated — every scored candidate carries the name of the reranker
that produced it, and app.query.score_gate refuses to compare a score
against a floor calibrated for a different one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.models.generator.base import Passage


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def rerank(
        self, question: str, candidates: Sequence[Passage], top_k: int = 5
    ) -> list[tuple[Passage, float]]: ...
