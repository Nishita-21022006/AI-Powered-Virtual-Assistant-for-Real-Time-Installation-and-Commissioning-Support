"""Orchestrates the ten query stages in order, against a latency budget.

    1. normalize        ~0ms    trim, detect language, reject empty/absurd
    2. cache lookup     ~20ms   hit returns immediately, skips 3-9
    3. rewrite          ~150ms  resolve follow-ups against session history
    4. metadata filter  ~0ms    this product, current versions only
    5. hybrid retrieve  ~200ms  dense + lexical, ~50 candidates
    6. rerank           ~250ms  cross-encoder, top 5
    7. score gate       ~0ms    below floor -> abstain, do not generate
    8. generate         ~1200ms answer from the 5 passages, with citations
    9. verify           ~50ms   deterministic checks, reject on failure
   10. respond          --      answer + document/section/page per claim

Stages 7 and 9 are the two places the pipeline is allowed to stop and
refuse. Total budget ~3s; generation dominates, which is why a stage 2
cache hit is worth so much.

DEGRADED MODE IS NOT AN ERROR PATH. With no generator configured the
pipeline runs stages 1-7 and returns the retrieved passages with their
citations. An integrator shown the right section of the right manual
has been helped. The system never returns nothing — so `answer` being
None is a normal outcome, and callers must render `passages` regardless
of whether `answer` is set.

That is also why this is wired before the generator exists: the
retrieval-only outcome is the bottom of the fallback chain
(generator API -> self-hosted -> retrieval-only), and it is the one
that must work without any credential at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from app.models.generator.api import GeneratorError
from app.models.generator.base import Passage
from app.query.normalize import NormalizationError, normalize
from app.query.rerank import RerankedHit, rerank
from app.query.retrieve import InMemoryIndex
from app.query.score_gate import RerankedCandidate, score_gate
from app.query.verify import SourcePassage, verify


class Outcome(StrEnum):
    ANSWERED = "answered"
    ABSTAINED_LOW_SCORE = "abstained_low_score"
    ABSTAINED_NO_RESULTS = "abstained_no_results"
    REJECTED_BY_VERIFICATION = "rejected_by_verification"
    RETRIEVAL_ONLY = "retrieval_only"
    REJECTED_INPUT = "rejected_input"


@dataclass
class QueryResult:
    outcome: Outcome
    question: str
    answer: str | None = None
    passages: list[RerankedHit] = field(default_factory=list)
    reason: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.outcome is Outcome.ANSWERED

    @property
    def has_citations(self) -> bool:
        return bool(self.passages)


class QueryPipeline:
    def __init__(
        self,
        index: InMemoryIndex,
        embedder,
        reranker=None,
        generator=None,
        score_floor: float = 0.5,
        retrieve_limit: int = 50,
        top_k: int = 5,
        show_sources_floor: float = -5.0,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator
        self.score_floor = score_floor
        # Below this, nothing in the corpus is genuinely related and we
        # say so rather than offering the nearest vectors. Like
        # score_floor it is reranker-specific; both need calibrating
        # against the evaluation set.
        self.show_sources_floor = show_sources_floor
        self.retrieve_limit = retrieve_limit
        self.top_k = top_k

    def run(self, question: str, documents: set[str] | None = None) -> QueryResult:
        timings: dict[str, float] = {}

        def timed(name: str, start: float) -> None:
            timings[name] = round((time.perf_counter() - start) * 1000, 1)

        t = time.perf_counter()
        try:
            query = normalize(question)
        except NormalizationError as exc:
            return QueryResult(
                outcome=Outcome.REJECTED_INPUT, question=question, reason=str(exc)
            )
        timed("normalize", t)

        t = time.perf_counter()
        vector = self.embedder.embed_query_sync(query)
        candidates = self.index.search(
            query, vector, limit=self.retrieve_limit, documents=documents
        )
        timed("retrieve", t)

        if not candidates:
            return QueryResult(
                outcome=Outcome.ABSTAINED_NO_RESULTS,
                question=query,
                reason="nothing in the indexed documentation matched this question",
                timings_ms=timings,
            )

        t = time.perf_counter()
        if self.reranker is not None:
            hits = rerank(query, candidates, self.reranker, top_k=self.top_k)
        else:
            # Without a reranker the retrieval order stands. Scores are
            # RRF values on a different scale from any cross-encoder's,
            # so they are tagged as such and the gate will refuse to
            # compare them against a reranker-calibrated floor.
            hits = [
                RerankedHit(
                    chunk_id=c.chunk.chunk_id,
                    document=c.chunk.document,
                    text=c.chunk.text,
                    score=c.score,
                    reranker="none:rrf",
                    page=c.chunk.page,
                    section=c.chunk.section,
                    has_warning=c.chunk.has_warning,
                    warnings=c.chunk.warnings,
                    retrieval_rank=i + 1,
                    reranked_rank=i + 1,
                )
                for i, c in enumerate(candidates[: self.top_k])
            ]
        timed("rerank", t)

        # The floor is calibrated against one reranker's score scale.
        # With no reranker the scores are RRF values — around 0.03 at
        # rank 1 — and comparing those against a cross-encoder floor
        # abstains on everything, including perfect hits. So without a
        # reranker there is no meaningful gate, and therefore no
        # permission to generate: unganged evidence must never reach the
        # model. Retrieval-only is the correct outcome, not an error.
        if self.reranker is None:
            return QueryResult(
                outcome=Outcome.RETRIEVAL_ONLY,
                question=query,
                passages=hits,
                reason=(
                    "no reranker configured, so the abstention floor is uncalibrated; "
                    "returning sources without a written answer"
                ),
                timings_ms=timings,
            )

        t = time.perf_counter()
        decision = score_gate(
            [
                RerankedCandidate(
                    chunk_id=h.chunk_id,
                    doc_id=h.document,
                    score=h.score,
                    reranker=h.reranker,
                    section=h.section,
                    page=h.page,
                    text=h.text,
                )
                for h in hits
            ],
            floor=self.score_floor,
            expected_reranker=self.reranker.name,
        )
        timed("score_gate", t)

        if not decision.proceed:
            # Two kinds of refusal, and conflating them is what makes an
            # abstention useless. Just below the floor means "related
            # material exists, read it yourself" — show it. Far below
            # means nothing in the corpus is on this topic at all, and
            # offering the nearest vectors anyway returns whatever is
            # most generic, which in a manual is the legal front matter.
            # Saying plainly that we have nothing is more honest and
            # more useful than five pages of copyright notice.
            best = decision.best_score
            nothing_close = best is None or best < self.show_sources_floor
            return QueryResult(
                outcome=Outcome.ABSTAINED_LOW_SCORE,
                question=query,
                passages=[] if nothing_close else hits,
                reason=(
                    "No documentation in the indexed manuals covers this. "
                    "This assistant only answers installation and commissioning "
                    "questions — for anything else, contact technical support."
                    if nothing_close
                    else "Not confident enough to answer. The most related "
                    "sections are below — please read them and verify."
                ),
                timings_ms=timings,
            )

        if self.generator is None:
            return QueryResult(
                outcome=Outcome.RETRIEVAL_ONLY,
                question=query,
                passages=hits,
                reason="no generator configured; returning sources without a written answer",
                timings_ms=timings,
            )

        t = time.perf_counter()
        try:
            generation = self.generator.generate_sync(
                query,
                [
                    Passage(
                        chunk_id=h.chunk_id,
                        doc_id=h.document,
                        text=h.text,
                        section=h.section,
                        page=h.page,
                        warnings=h.warnings,
                    )
                    for h in hits
                ],
            )
        except GeneratorError as exc:
            # Bottom of the fallback chain. A provider outage or a
            # rate limit must not cost the user the sources we already
            # found — the system never returns nothing.
            timed("generate", t)
            return QueryResult(
                outcome=Outcome.RETRIEVAL_ONLY,
                question=query,
                passages=hits,
                reason=f"generator unavailable ({exc}); returning sources only",
                timings_ms=timings,
            )
        timed("generate", t)

        if generation.abstained:
            return QueryResult(
                outcome=Outcome.ABSTAINED_LOW_SCORE,
                question=query,
                passages=hits,
                reason=(
                    generation.text
                    or "the retrieved passages do not contain the answer"
                ),
                timings_ms=timings,
            )

        t = time.perf_counter()
        # The warning requirement attaches to the top-ranked passage only.
        #
        # Five passages are retrieved and any of them may carry a warning
        # about something the answer never mentions — a cable test voltage
        # surfacing alongside a question about a USB port. Demanding every
        # warning in the retrieved set be recited rejects correct answers
        # for failing to mention an unrelated hazard, and an installer who
        # is shown irrelevant warnings on every answer stops reading them.
        #
        # Rank 1 is the passage the answer is overwhelmingly drawn from, so
        # that is where the obligation sits. The looser edge: a procedure
        # assembled from ranks 1 and 2 could drop rank 2's warning. The
        # chunker is the real defence there — it keeps a warning in the
        # same chunk as its procedure — and this is the backstop.
        verdict = verify(
            generation.text,
            [
                SourcePassage(
                    doc_id=h.document,
                    text=h.text,
                    section=h.section,
                    page=h.page,
                    warnings=h.warnings if i == 0 else (),
                )
                for i, h in enumerate(hits)
            ],
        )
        timed("verify", t)

        if not verdict.accepted:
            # Tier 3 rejected it. The answer is discarded entirely — a
            # partially-trustworthy answer is not a category we offer —
            # but the sources still go back.
            return QueryResult(
                outcome=Outcome.REJECTED_BY_VERIFICATION,
                question=query,
                passages=hits,
                reason="; ".join(verdict.details) or "failed output verification",
                timings_ms=timings,
            )

        return QueryResult(
            outcome=Outcome.ANSWERED,
            question=query,
            answer=generation.text,
            passages=hits,
            timings_ms=timings,
        )
