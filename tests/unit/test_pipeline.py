"""The query pipeline, with stubs for the model layer.

The case that matters most here is degraded mode. With no generator the
pipeline must still return the right passages with their citations —
that is the bottom of the fallback chain and the only rung that works
with no credential at all. A regression that turned it into an error
would take the system from "helped the integrator" to "returned
nothing", which the architecture forbids outright.
"""

from __future__ import annotations

import pytest

from app.query.pipeline import Outcome, QueryPipeline
from app.query.retrieve import IndexedChunk, InMemoryIndex

MOUNTING = "2.2 Wall mounting. Maximum screw head height is 3 mm."
TORQUE = "Tighten the four mounting screws to 2.5 Nm."
UNRELATED = "Appendix C: disposal and recycling information."


class StubEmbedder:
    """Maps a handful of known strings to fixed vectors so retrieval is
    deterministic and the test is about the pipeline, not the model."""

    name = "stub-embedder"
    dim = 2

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def embed_query_sync(self, text):
        for key, vec in self.mapping.items():
            if key in text.casefold():
                return vec
        return [1.0, 0.0]


class StubReranker:
    name = "stub-reranker"

    def rerank_sync(self, question, texts, top_k=None):
        def score(i):
            return 0.9 if "screw head" in texts[i] else 0.1

        order = sorted(range(len(texts)), key=lambda i: -score(i))
        if top_k is not None:
            order = order[:top_k]
        return [(i, score(i)) for i in order]


@pytest.fixture
def index():
    idx = InMemoryIndex()
    idx.add(
        [
            IndexedChunk("c0", "MAN-1.pdf", MOUNTING, page=8, section="2.2", has_warning=True),
            IndexedChunk("c1", "MAN-1.pdf", TORQUE, page=8, section="2.2"),
            IndexedChunk("c2", "MAN-1.pdf", UNRELATED, page=40, section="C"),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    return idx


def test_degraded_mode_returns_passages_with_citations_and_no_answer(index):
    """No generator configured. This is a normal outcome, not an error."""
    result = QueryPipeline(index, StubEmbedder(), reranker=None, generator=None).run(
        "maximum screw head height"
    )

    assert result.outcome is Outcome.RETRIEVAL_ONLY
    assert result.answer is None
    assert result.passages, "degraded mode returned nothing — the one forbidden outcome"
    assert result.has_citations
    top = result.passages[0]
    assert top.document == "MAN-1.pdf"
    assert top.page == 8


def test_reranker_reorders_before_the_gate(index):
    pipeline = QueryPipeline(
        index, StubEmbedder(), reranker=StubReranker(), score_floor=0.5
    )
    result = pipeline.run("how high can the screw head be")
    assert result.passages[0].chunk_id == "c0"
    assert result.passages[0].reranker == "stub-reranker"


def test_below_floor_abstains_but_still_shows_the_closest_sections(index):
    """Abstention is a feature: refuse to answer, but never refuse to
    help. The closest sections go back regardless."""
    pipeline = QueryPipeline(
        index, StubEmbedder(), reranker=StubReranker(), score_floor=0.99
    )
    result = pipeline.run("something only weakly related")

    assert result.outcome is Outcome.ABSTAINED_LOW_SCORE
    assert result.answer is None
    assert result.passages, "abstention must still return the closest sections"


def test_empty_index_abstains_rather_than_raising():
    pipeline = QueryPipeline(InMemoryIndex(), StubEmbedder(), reranker=None)
    result = pipeline.run("anything at all")
    assert result.outcome is Outcome.ABSTAINED_NO_RESULTS
    assert result.passages == []


@pytest.mark.parametrize("bad", ["", "   ", "?", None])
def test_junk_input_is_rejected_at_tier_zero(index, bad):
    result = QueryPipeline(index, StubEmbedder()).run(bad)
    assert result.outcome is Outcome.REJECTED_INPUT
    assert result.reason


def test_metadata_filter_restricts_to_the_named_documents(index):
    pipeline = QueryPipeline(index, StubEmbedder(), reranker=None)
    assert pipeline.run("screw", documents={"OTHER.pdf"}).outcome is (
        Outcome.ABSTAINED_NO_RESULTS
    )


def test_timings_are_recorded_for_each_stage(index):
    result = QueryPipeline(index, StubEmbedder(), reranker=StubReranker()).run("screw head")
    assert {"normalize", "retrieve", "rerank", "score_gate"} <= set(result.timings_ms)
    assert all(v >= 0 for v in result.timings_ms.values())


def test_scores_without_a_reranker_are_tagged_as_a_different_scale(index):
    """RRF scores are not cross-encoder scores. The gate must be able to
    tell, or a floor calibrated for one silently applies to the other."""
    result = QueryPipeline(index, StubEmbedder(), reranker=None, score_floor=0.0).run(
        "screw head"
    )
    assert all(p.reranker == "none:rrf" for p in result.passages)
