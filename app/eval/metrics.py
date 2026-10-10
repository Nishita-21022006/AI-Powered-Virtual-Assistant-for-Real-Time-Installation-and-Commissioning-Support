"""Metrics.

Phase 1 (retrieval only): recall@k and MRR against the expected source —
did the right passage come back at all, and how high.
Phase 2: the same, before and after reranking, so the reranker's latency
cost is justified by a measured delta rather than assumed.
Phase 3 onward: answer accuracy, abstention rate, and — separately —
false-answer rate, which is the number that actually matters. The target
shape is ~85% answered at ~99% accurate, not maximum coverage.

A retrieved chunk counts as a hit when it comes from the expected
document AND contains the question's verbatim supporting quote. Matching
on document alone would score a hit for retrieving the right manual and
the wrong page, which in a 139-page document is not a hit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document: str
    text: str
    page: int | None = None
    section: str | None = None


def is_hit(chunk: RetrievedChunk, document: str, supporting_quote: str) -> bool:
    if _normalise(chunk.document) != _normalise(document):
        return False
    return _normalise(supporting_quote) in _normalise(chunk.text)


def rank_of_hit(
    retrieved: Sequence[RetrievedChunk], document: str, supporting_quote: str
) -> int | None:
    """1-indexed rank of the first hit, or None if the quote never came back."""
    for rank, chunk in enumerate(retrieved, start=1):
        if is_hit(chunk, document, supporting_quote):
            return rank
    return None


@dataclass(frozen=True)
class RetrievalResult:
    question_id: str
    rank: int | None

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.rank is None else 1.0 / self.rank


@dataclass(frozen=True)
class RetrievalReport:
    results: list[RetrievalResult]

    def recall_at(self, k: int) -> float:
        if not self.results:
            return 0.0
        return sum(r.hit_at(k) for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def misses(self) -> list[str]:
        return [r.question_id for r in self.results if r.rank is None]

    def summary(self, ks: Sequence[int] = (1, 3, 5, 10, 20, 50)) -> dict[str, float]:
        report = {f"recall@{k}": self.recall_at(k) for k in ks}
        report["mrr"] = self.mrr
        return report


@dataclass(frozen=True)
class AnswerOutcome:
    question_id: str
    answered: bool
    correct: bool | None
    should_have_answered: bool


@dataclass(frozen=True)
class AnswerReport:
    """The shape that matters: high accuracy on what is answered, and
    refusals concentrated where refusal was correct.

    false_answer_rate is the headline risk number — answers given that
    were wrong, or given at all when the question was out of scope. One
    confidently wrong wiring answer is worse than fifty honest refusals,
    so this is tracked separately from accuracy rather than folded into it.
    """

    outcomes: list[AnswerOutcome]

    def _subset(self, in_scope: bool) -> list[AnswerOutcome]:
        return [o for o in self.outcomes if o.should_have_answered is in_scope]

    @property
    def answer_rate(self) -> float:
        scoped = self._subset(True)
        if not scoped:
            return 0.0
        return sum(o.answered for o in scoped) / len(scoped)

    @property
    def accuracy(self) -> float:
        answered = [o for o in self._subset(True) if o.answered]
        if not answered:
            return 0.0
        return sum(bool(o.correct) for o in answered) / len(answered)

    @property
    def abstention_precision(self) -> float:
        """Of the questions that should have been refused, how many were."""
        out_of_scope = self._subset(False)
        if not out_of_scope:
            return 0.0
        return sum(not o.answered for o in out_of_scope) / len(out_of_scope)

    @property
    def false_answer_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        wrong = [
            o
            for o in self.outcomes
            if o.answered and (not o.should_have_answered or o.correct is False)
        ]
        return len(wrong) / len(self.outcomes)

    def summary(self) -> dict[str, float]:
        return {
            "answer_rate": self.answer_rate,
            "accuracy": self.accuracy,
            "abstention_precision": self.abstention_precision,
            "false_answer_rate": self.false_answer_rate,
        }
