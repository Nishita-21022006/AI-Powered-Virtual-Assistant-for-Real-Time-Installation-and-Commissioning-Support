"""Hybrid retrieval: fusion, the lexical path, and metadata filtering.

Uses synthetic vectors so the ranking logic is tested independently of
any embedding model — a failure here is a bug in retrieval, not a bad
embedding. The case that matters most is the exact-string one: a part
number or fault code that dense search treats as noise must still come
back, which is the entire reason the lexical half exists (failure
mode 2).
"""

from __future__ import annotations

import numpy as np

from app.query.retrieve import BM25, IndexedChunk, InMemoryIndex, tokenize


def _chunk(cid, text, document="MAN-1.pdf", page=1):
    return IndexedChunk(chunk_id=cid, document=document, text=text, page=page)


def _orthogonal(n, dim=8):
    """One distinct unit vector per chunk, so dense similarity is fully
    controlled by the test rather than by text content."""
    vectors = []
    for i in range(n):
        v = [0.0] * dim
        v[i % dim] = 1.0
        vectors.append(v)
    return vectors


def test_tokenizer_keeps_part_codes_intact():
    assert tokenize("Order board E-412 now") == ["order", "board", "e-412", "now"]
    assert tokenize("See 89470K2.GB0 page 6") == ["see", "89470k2.gb0", "page", "6"]
    assert "sem+" not in tokenize("SEM+ terminal")  # '+' is not a token char
    assert tokenize("SEM+ terminal") == ["sem", "terminal"]


def test_bm25_ranks_the_chunk_containing_the_rare_term_first():
    texts = [
        "General mounting instructions for the housing.",
        "Fault code E-412 indicates a failed control board.",
        "General wiring instructions for the housing.",
    ]
    scores = BM25().fit(texts).scores("E-412")
    assert int(np.argmax(scores)) == 1
    assert scores[0] == 0.0 and scores[2] == 0.0


def test_exact_part_code_is_retrieved_although_dense_similarity_is_worst():
    """The failure-mode-2 case. The chunk holding the part code is given
    the LEAST similar dense vector; the lexical half must still surface
    it, or a part-number query returns nothing useful."""
    chunks = [
        _chunk("c0", "Mounting the panel on a wall with four screws."),
        _chunk("c1", "Tightening torque for the terminal screws is 2.5 Nm."),
        _chunk("c2", "Replacement control board part number E-412."),
    ]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    index = InMemoryIndex()
    index.add(chunks, vectors)

    # Query vector points away from c2 entirely - dense alone ranks it last.
    hits = index.search("part number E-412", [1.0, 0.0], limit=3)
    ids = [h.chunk.chunk_id for h in hits]

    assert ids[0] == "c2", f"lexical half failed to surface the part code: {ids}"
    assert next(h for h in hits if h.chunk.chunk_id == "c2").lexical_rank == 1


def test_dense_half_finds_a_paraphrase_with_no_shared_rare_terms():
    chunks = [
        _chunk("c0", "Fault code E-412 indicates a failed control board."),
        _chunk("c1", "Affix the unit to a vertical surface using four fixings."),
    ]
    # Query vector aligned with c1.
    index = InMemoryIndex()
    index.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    hits = index.search("how do I mount it on the wall", [0.0, 1.0], limit=2)
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].dense_rank == 1


def test_fusion_promotes_a_chunk_both_engines_like():
    chunks = [
        _chunk("c0", "Tightening torque for the mounting screws is 2.5 Nm."),
        _chunk("c1", "Torque values are listed in the appendix."),
        _chunk("c2", "Unrelated commissioning notes."),
    ]
    index = InMemoryIndex()
    index.add(chunks, [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    hits = index.search("mounting screws torque", [1.0, 0.0], limit=3)
    top = hits[0]
    assert top.chunk.chunk_id == "c0"
    assert top.dense_rank is not None and top.lexical_rank is not None


def test_metadata_filter_excludes_other_documents():
    chunks = [
        _chunk("a0", "Torque is 2.5 Nm.", document="A.pdf"),
        _chunk("b0", "Torque is 4.0 Nm.", document="B.pdf"),
    ]
    index = InMemoryIndex()
    index.add(chunks, [[1.0, 0.0], [1.0, 0.0]])

    hits = index.search("torque", [1.0, 0.0], limit=5, documents={"B.pdf"})
    assert [h.chunk.document for h in hits] == ["B.pdf"]


def test_empty_index_returns_nothing_rather_than_raising():
    assert InMemoryIndex().search("anything", [1.0, 0.0]) == []


def test_mismatched_chunk_and_vector_counts_is_an_error():
    index = InMemoryIndex()
    try:
        index.add([_chunk("c0", "text")], [])
    except ValueError as exc:
        assert "1 chunks but 0 vectors" in str(exc)
    else:
        raise AssertionError("silently accepted a mismatched index")


def test_limit_is_respected():
    n = 30
    chunks = [_chunk(f"c{i}", f"chunk number {i} about mounting") for i in range(n)]
    index = InMemoryIndex()
    index.add(chunks, _orthogonal(n))
    assert len(index.search("mounting", [1.0] + [0.0] * 7, limit=5)) == 5
