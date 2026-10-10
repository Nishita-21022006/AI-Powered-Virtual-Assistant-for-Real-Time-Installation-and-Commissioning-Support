"""Reranker — BGE reranker, m3 family, local, CPU, free, ~1-2 GB.

Pairs with the BGE-M3 embedder. A cross-encoder reads the question
against each candidate together, rather than comparing two independently
produced vectors, which is why it is more accurate than retrieval and
too slow to run over the whole corpus. Hence the shape of the pipeline:
retrieve ~50 cheaply, rerank those to 5.

The biggest single accuracy gain available, per the architecture — and
phase 2 exists to measure that claim rather than assume it.

Its score scale is what the abstention floor is calibrated against, so
every scored candidate is tagged with `name`. app.query.score_gate
refuses to compare a score against a floor calibrated for a different
reranker rather than silently shifting the abstention rate.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

MAX_LENGTH = 512
"""Token cap per (question, passage) pair.

This is load-bearing for latency, not a detail. The m3 family accepts
8192 tokens and sentence-transformers defaults to the model maximum, so
leaving it unset lets a long chunk produce a very long sequence — and
attention cost is quadratic in that length. Measured on this corpus,
the unset default made a 50-candidate rerank take tens of seconds per
query against a ~250ms budget.

512 covers a chunk's opening comfortably, which is where a section
heading and its first lines sit. If recall at rank 5 ever turns out to
depend on text past that point, raise this and re-measure rather than
guessing — it trades latency directly.
"""


class LocalReranker:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        max_length: int = MAX_LENGTH,
    ):
        self.name = model
        self.max_length = max_length
        self._cache_dir = cache_dir or os.environ.get("MODEL_CACHE_DIR") or None

    @cached_property
    def _model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            self.name,
            cache_folder=self._cache_dir,
            device="cpu",
            max_length=self.max_length,
        )

    def rerank_sync(
        self, question: str, texts: Sequence[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Returns (original index, score), best first.

        Indices rather than the passages themselves so the caller keeps
        ownership of its own objects and nothing is copied or reordered
        behind its back.
        """
        if not texts:
            return []
        scores = self._model.predict(
            [(question, t) for t in texts], show_progress_bar=False
        )
        order = sorted(range(len(texts)), key=lambda i: -float(scores[i]))
        if top_k is not None:
            order = order[:top_k]
        return [(i, float(scores[i])) for i in order]

    async def rerank(
        self, question: str, texts: Sequence[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        return self.rerank_sync(question, texts, top_k)
