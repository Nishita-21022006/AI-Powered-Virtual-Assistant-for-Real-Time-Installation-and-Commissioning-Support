"""Stage 7 — the abstention gate. Guardrail tier 1.

If the best reranked score is below the configured floor, stop here. Do
not generate. Return the closest related sections, name the document that
would cover the question, and offer the support path.

This is a threshold, not a model judgment, and it is the mitigation for
the most common production RAG failure: a model handed five weakly
related passages will cheerfully invent an answer.

The floor is `RETRIEVAL_SCORE_FLOOR` from config, and the caller passes it
in -- this module reads no settings and touches no I/O, so it is testable
without an environment.

CALIBRATION IS RERANKER-SPECIFIC. A Cohere Rerank score and a BGE
cross-encoder score are not the same quantity, and a floor calibrated
against one is meaningless against the other. So every candidate carries
the id of the reranker that scored it, the decision records which
reranker it gated on, and a candidate set that disagrees with the
configured reranker -- or disagrees with itself -- abstains rather than
comparing an uncalibrated number against the floor. Swapping rerankers
therefore fails closed and loudly instead of silently shifting the
abstention rate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

ABSTENTION_PASSAGE_COUNT = 5
"""How many of the closest sections to hand to escalation when abstaining."""


@dataclass(frozen=True)
class RerankedCandidate:
    chunk_id: str
    doc_id: str
    score: float
    reranker: str
    section: str | None = None
    page: int | None = None
    text: str = ""


class GateOutcome(StrEnum):
    PROCEED = "proceed"
    NO_CANDIDATES = "no_candidates"
    BELOW_FLOOR = "below_floor"
    UNUSABLE_SCORE = "unusable_score"
    RERANKER_MISMATCH = "reranker_mismatch"
    UNUSABLE_FLOOR = "unusable_floor"


@dataclass(frozen=True)
class GateDecision:
    proceed: bool
    outcome: GateOutcome
    floor: float
    reranker: str | None
    best_score: float | None
    candidates: tuple[RerankedCandidate, ...]
    detail: str = ""

    def __bool__(self) -> bool:
        return self.proceed


def score_gate(
    candidates: Sequence[RerankedCandidate],
    floor: float,
    expected_reranker: str | None = None,
) -> GateDecision:
    """Decide whether the reranked evidence is strong enough to generate from.

    `expected_reranker` is the reranker the floor was calibrated against
    (config). When given, candidates scored by anything else abstain: the
    floor does not transfer across score scales.

    Returns PROCEED only when a finite best score is at or above the floor.
    Every other path abstains, including the ones that look like bugs
    rather than weak evidence -- a NaN score or a mixed candidate set means
    the comparison against the floor is not meaningful, and an unmeaningful
    comparison must never be resolved in favour of generating.
    """
    ranked = tuple(sorted(candidates, key=lambda c: c.score, reverse=True))
    closest = ranked[:ABSTENTION_PASSAGE_COUNT]

    if not ranked:
        return GateDecision(
            False, GateOutcome.NO_CANDIDATES, floor, expected_reranker, None, (),
            "retrieval returned nothing to score",
        )

    rerankers = {c.reranker for c in ranked}
    reranker = ranked[0].reranker
    if len(rerankers) > 1:
        return GateDecision(
            False, GateOutcome.RERANKER_MISMATCH, floor, None, None, closest,
            "candidates scored by more than one reranker: " + ", ".join(sorted(rerankers)),
        )
    if expected_reranker is not None and reranker != expected_reranker:
        return GateDecision(
            False, GateOutcome.RERANKER_MISMATCH, floor, reranker, None, closest,
            "floor calibrated for " + expected_reranker + ", scores produced by " + reranker,
        )

    if not isinstance(floor, (int, float)) or not math.isfinite(floor):
        return GateDecision(
            False, GateOutcome.UNUSABLE_FLOOR, floor, reranker, None, closest,
            "configured score floor is not a finite number: " + repr(floor),
        )

    # Checked across the whole set, not just the top candidate: a NaN sorts
    # arbitrarily, so a NaN anywhere means the ordering itself is untrustworthy.
    unusable = [c for c in ranked if not math.isfinite(c.score)]
    if unusable:
        return GateDecision(
            False, GateOutcome.UNUSABLE_SCORE, floor, reranker, None, closest,
            "non-finite candidate score: " + ", ".join(repr(c.score) for c in unusable),
        )

    best = ranked[0]

    if best.score < floor:
        return GateDecision(
            False, GateOutcome.BELOW_FLOOR, floor, reranker, best.score, closest,
            f"best reranked score {best.score!r} is below the floor {floor!r}",
        )

    return GateDecision(True, GateOutcome.PROCEED, floor, reranker, best.score, closest)
