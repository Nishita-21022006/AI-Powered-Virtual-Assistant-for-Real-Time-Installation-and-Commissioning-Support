"""The abstention gate.

Below the floor the pipeline must stop before generation and return the
closest sections instead. Abstention is the feature; a test that lets a
weak-evidence query through to the generator is a failing test.
"""

import dataclasses
import math

import pytest

from app.query.score_gate import (
    ABSTENTION_PASSAGE_COUNT,
    GateOutcome,
    RerankedCandidate,
    score_gate,
)

RERANKER = "cohere/rerank-v3.5"
FLOOR = 0.5


def candidate(score, reranker=RERANKER, chunk_id=None):
    return RerankedCandidate(
        chunk_id=chunk_id or ("chunk-" + str(score)),
        doc_id="INST-4412",
        score=score,
        reranker=reranker,
        section="4.2",
        page=12,
        text="Tighten the four M6 mounting screws to 2.5 Nm.",
    )


def test_proceeds_when_the_best_score_clears_the_floor():
    decision = score_gate([candidate(0.81), candidate(0.42)], FLOOR, RERANKER)
    assert decision.proceed
    assert decision.outcome is GateOutcome.PROCEED
    assert decision.best_score == 0.81
    assert bool(decision) is True


def test_abstains_when_every_score_is_below_the_floor():
    decision = score_gate([candidate(0.49), candidate(0.12)], FLOOR, RERANKER)
    assert not decision.proceed
    assert decision.outcome is GateOutcome.BELOW_FLOOR
    assert decision.best_score == 0.49


def test_a_score_exactly_on_the_floor_proceeds():
    assert score_gate([candidate(FLOOR)], FLOOR, RERANKER).proceed


def test_a_score_a_hair_under_the_floor_abstains():
    decision = score_gate([candidate(FLOOR - 1e-9)], FLOOR, RERANKER)
    assert decision.outcome is GateOutcome.BELOW_FLOOR


def test_many_weak_candidates_never_add_up_to_strong_evidence():
    decision = score_gate([candidate(0.3, chunk_id="c" + str(i)) for i in range(50)], FLOOR, RERANKER)
    assert not decision.proceed
    assert decision.outcome is GateOutcome.BELOW_FLOOR


def test_empty_candidate_set_abstains():
    decision = score_gate([], FLOOR, RERANKER)
    assert not decision.proceed
    assert decision.outcome is GateOutcome.NO_CANDIDATES
    assert decision.candidates == ()
    assert decision.best_score is None


def test_the_best_score_is_found_regardless_of_input_order():
    decision = score_gate([candidate(0.2), candidate(0.9), candidate(0.4)], FLOOR, RERANKER)
    assert decision.proceed
    assert decision.best_score == 0.9


def test_the_decision_records_the_reranker_that_produced_the_scores():
    decision = score_gate([candidate(0.81)], FLOOR, RERANKER)
    assert decision.reranker == RERANKER
    assert decision.floor == FLOOR


def test_scores_from_an_unexpected_reranker_abstain_rather_than_reuse_the_calibration():
    decision = score_gate([candidate(0.99, reranker="bge-reranker-v2-m3")], FLOOR, RERANKER)
    assert not decision.proceed
    assert decision.outcome is GateOutcome.RERANKER_MISMATCH
    assert "bge-reranker-v2-m3" in decision.detail


def test_a_candidate_set_scored_by_two_rerankers_abstains():
    decision = score_gate(
        [candidate(0.99), candidate(0.98, reranker="bge-reranker-v2-m3")], FLOOR, RERANKER
    )
    assert not decision.proceed
    assert decision.outcome is GateOutcome.RERANKER_MISMATCH
    assert decision.reranker is None


def test_without_an_expected_reranker_the_candidates_own_id_is_recorded():
    decision = score_gate([candidate(0.81, reranker="bge-reranker-v2-m3")], FLOOR)
    assert decision.proceed
    assert decision.reranker == "bge-reranker-v2-m3"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_score_abstains(bad):
    decision = score_gate([candidate(bad), candidate(0.9)], FLOOR, RERANKER)
    assert not decision.proceed
    assert decision.outcome is GateOutcome.UNUSABLE_SCORE


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_floor_abstains(bad):
    decision = score_gate([candidate(0.99)], bad, RERANKER)
    assert not decision.proceed
    assert decision.outcome is GateOutcome.UNUSABLE_FLOOR


def test_a_floor_of_zero_is_honoured_and_not_mistaken_for_unset():
    assert score_gate([candidate(0.01)], 0.0, RERANKER).proceed


def test_a_floor_above_every_possible_score_abstains_on_everything():
    assert not score_gate([candidate(0.999)], 1.0, RERANKER).proceed


def test_abstention_still_returns_the_closest_sections_for_escalation():
    weak = [candidate(0.01 * i, chunk_id="c" + str(i)) for i in range(1, 9)]
    decision = score_gate(weak, FLOOR, RERANKER)
    assert not decision.proceed
    assert len(decision.candidates) == ABSTENTION_PASSAGE_COUNT
    scores = [c.score for c in decision.candidates]
    assert scores == sorted(scores, reverse=True)
    assert decision.candidates[0].doc_id == "INST-4412"


def test_proceeding_hands_on_the_top_candidates_in_rank_order():
    decision = score_gate(
        [candidate(0.4), candidate(0.95), candidate(0.6)], FLOOR, RERANKER
    )
    assert [c.score for c in decision.candidates] == [0.95, 0.6, 0.4]


def test_a_reranker_mismatch_still_surfaces_sections_to_show():
    decision = score_gate([candidate(0.99, reranker="other")], FLOOR, RERANKER)
    assert len(decision.candidates) == 1
    assert decision.best_score is None


def test_the_gate_does_not_mutate_or_reorder_its_input():
    given = [candidate(0.2), candidate(0.9)]
    snapshot = list(given)
    score_gate(given, FLOOR, RERANKER)
    assert given == snapshot


def test_the_decision_is_immutable():
    decision = score_gate([candidate(0.81)], FLOOR, RERANKER)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.proceed = False
