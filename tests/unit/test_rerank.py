"""Stage 6 plumbing: that reranking reorders candidates, truncates to
top_k, and carries the reranker's identity onto every hit.

A stub reranker stands in for the cross-encoder so these test the stage,
not the model. Whether reranking actually improves ranking is a
different question, measured against the evaluation set in
app.eval.run — a unit test cannot answer it and should not pretend to.
"""

from __future__ import annotations

from app.query.rerank import rerank
from app.query.retrieve import IndexedChunk, Scored


class StubReranker:
    """Scores by position in a supplied preference list, so the test
    controls the reordering exactly."""

    name = "stub-reranker-v1"

    def __init__(self, preferred: list[str]):
        self.preferred = preferred

    def rerank_sync(self, question, texts, top_k=None):
        def score(i: int) -> float:
            text = texts[i]
            for rank, token in enumerate(self.preferred):
                if token in text:
                    return 10.0 - rank
            return 0.0

        order = sorted(range(len(texts)), key=lambda i: -score(i))
        if top_k is not None:
            order = order[:top_k]
        return [(i, score(i)) for i in order]


def _candidates(*texts: str) -> list[Scored]:
    return [
        Scored(
            chunk=IndexedChunk(
                chunk_id=f"c{i}",
                document="MAN-1.pdf",
                text=t,
                page=i + 1,
                has_warning=("WARNING" in t),
            ),
            score=1.0 / (i + 1),
            dense_rank=i + 1,
        )
        for i, t in enumerate(texts)
    ]


def test_reranking_promotes_the_better_passage_from_a_low_retrieval_rank():
    candidates = _candidates(
        "General notes about the housing.",
        "More general notes.",
        "Maximum screw head height is 3 mm.",
    )
    hits = rerank("screw head height", candidates, StubReranker(["screw head"]), top_k=3)

    assert hits[0].chunk_id == "c2"
    assert hits[0].retrieval_rank == 3, "should record where it came from"
    assert hits[0].reranked_rank == 1


def test_top_k_truncates():
    candidates = _candidates(*[f"passage {i}" for i in range(20)])
    hits = rerank("anything", candidates, StubReranker([]), top_k=5)
    assert len(hits) == 5
    assert [h.reranked_rank for h in hits] == [1, 2, 3, 4, 5]


def test_every_hit_carries_the_reranker_name():
    """Load-bearing: the abstention floor is calibrated per reranker and
    score_gate refuses to compare across scales."""
    hits = rerank("q", _candidates("a", "b"), StubReranker(["a"]), top_k=2)
    assert {h.reranker for h in hits} == {"stub-reranker-v1"}


def test_chunk_metadata_survives_reranking():
    candidates = _candidates("WARNING: isolate the supply before opening.")
    hit = rerank("q", candidates, StubReranker([]), top_k=1)[0]
    assert hit.document == "MAN-1.pdf"
    assert hit.page == 1
    assert hit.has_warning is True


def test_empty_candidate_list_returns_empty_without_calling_the_model():
    class Exploding:
        name = "should-not-be-called"

        def rerank_sync(self, *a, **k):
            raise AssertionError("model invoked on an empty candidate list")

    assert rerank("q", [], Exploding(), top_k=5) == []


def test_scores_are_the_rerankers_not_the_retrievers():
    candidates = _candidates("alpha", "beta")
    hits = rerank("q", candidates, StubReranker(["beta"]), top_k=2)
    assert hits[0].score == 10.0
    assert hits[0].score != candidates[0].score
