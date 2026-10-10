"""The hard rule, under test: a safety warning is never separated from
the procedure it governs.

Given a section containing numbered steps and an interleaved warning,
chunking must not emit a chunk holding the step without its warning.
This is the test the architecture explicitly asks for, and it protects
against the failure that silently destroys equipment.

The synthetic cases pin the rule precisely. The corpus case guards
against the rule holding only on fixtures: in these manuals a warning
carries no "WARNING:" text at all — it is a pictogram — so a chunker
that passed on invented input with the word "WARNING" in it would
detect nothing on the real documents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunk import MAX_CHUNK_CHARS, Chunk, chunk_document
from app.ingestion.parse import Block, ParsedDocument, parse_pdf

CORPUS = Path(__file__).resolve().parents[2] / "Documents"
WARNED_MANUAL = CORPUS / "89475A3.GB0.pdf"


def _block(type_, text, page=1, top=0.0, **kw):
    return Block(type=type_, text=text, page=page, bbox=(0.0, top, 500.0, top + 10), **kw)


def _doc(blocks) -> ParsedDocument:
    return ParsedDocument(
        path=Path("synthetic.pdf"),
        page_count=1,
        title="synthetic",
        product=None,
        doc_number=None,
        revision=None,
        part_number=None,
        blocks=blocks,
    )


WARNING_TEXT = "Maximum screw head height is 3 mm, otherwise the electronics may be damaged."


def _mounting_section(step_filler: str = "") -> list[Block]:
    return [
        _block("heading", "Installation on a wall", top=0, level=2, section_number="2.2"),
        _block("paragraph", "Step 1: Mark the four hole positions." + step_filler, top=10),
        _block("paragraph", "Step 2: Drill and fit the wall plugs." + step_filler, top=20),
        _block("paragraph", "Step 3: Align the panel." + step_filler, top=30),
        _block("callout", WARNING_TEXT, top=40, callout_kind="warning"),
        _block("paragraph", "Step 4: Tighten the four mounting screws." + step_filler, top=50),
        _block("paragraph", "Step 5: Fit the terminal into the housing." + step_filler, top=60),
    ]


def _chunks_with_steps(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if "Step " in c.text]


def test_small_section_keeps_steps_and_warning_in_one_chunk():
    chunks = chunk_document(_doc(_mounting_section()))
    assert len(chunks) == 1
    assert "Step 4" in chunks[0].text
    assert WARNING_TEXT in chunks[0].text
    assert chunks[0].has_warning


def test_every_chunk_with_a_step_carries_the_warning_even_when_split():
    # Force a split: each step alone exceeds the chunk budget.
    filler = " " + ("padding " * (MAX_CHUNK_CHARS // 8))
    chunks = chunk_document(_doc(_mounting_section(filler)))

    assert len(chunks) > 1, "test needs an actually-split section to be meaningful"
    stepped = _chunks_with_steps(chunks)
    assert stepped, "split lost the steps entirely"
    for chunk in stepped:
        assert WARNING_TEXT in chunk.text, (
            f"chunk {chunk.ordinal} holds a step without its warning:\n{chunk.text[:300]}"
        )
        assert chunk.has_warning


def test_step_four_is_never_alone_in_a_chunk_without_the_warning():
    filler = " " + ("padding " * (MAX_CHUNK_CHARS // 8))
    for chunk in chunk_document(_doc(_mounting_section(filler))):
        if "Step 4" in chunk.text:
            assert WARNING_TEXT in chunk.text


def test_section_without_warnings_is_not_marked_as_warned():
    chunks = chunk_document(
        _doc(
            [
                _block("heading", "Technical data", top=0, level=2, section_number="8"),
                _block("paragraph", "Operating voltage 24 V DC.", top=10),
            ]
        )
    )
    assert len(chunks) == 1
    assert not chunks[0].has_warning
    assert chunks[0].warnings == ()


def test_warning_blocks_are_recorded_on_the_chunk_not_only_inlined():
    chunks = chunk_document(_doc(_mounting_section()))
    assert chunks[0].warnings == (WARNING_TEXT,)


def test_sections_are_not_split_at_a_fixed_word_count():
    """Two short adjacent sections must not be merged into one chunk, and
    a short section must not be broken up — the unit is the section."""
    chunks = chunk_document(
        _doc(
            [
                _block("heading", "First", top=0, level=2, section_number="1"),
                _block("paragraph", "Content of the first section.", top=10),
                _block("heading", "Second", top=20, level=2, section_number="2"),
                _block("paragraph", "Content of the second section.", top=30),
            ]
        )
    )
    assert len(chunks) == 2
    assert chunks[0].section_number == "1"
    assert chunks[1].section_number == "2"
    assert "second section" not in chunks[0].text


def test_chunk_boundaries_are_contiguous_and_non_overlapping():
    chunks = chunk_document(_doc(_mounting_section()))
    for previous, nxt in zip(chunks, chunks[1:]):
        assert previous.char_end == nxt.char_start
        assert previous.ordinal < nxt.ordinal


@pytest.mark.skipif(not WARNED_MANUAL.exists(), reason="corpus not present")
def test_real_manual_warnings_survive_chunking():
    """In this corpus a warning is a pictogram and carries no 'WARNING:'
    string, so a keyword-based chunker would silently find none. Assert
    we detect warnings at all, and that each one reaches a chunk."""
    from app.ingestion.chunk import is_boilerplate

    doc = parse_pdf(WARNED_MANUAL, extract_figure_images=False)
    # Two kinds of block carry a warning pictogram without governing any
    # procedure, and both live on the front matter that ingestion drops:
    # the symbol legend ("Warning - Designates risks to person...") which
    # defines what the pictogram means, and the WEEE disposal notice,
    # which is a legal statement about end-of-life recycling.
    #
    # Losing those is correct. The rule protects warnings attached to
    # steps an installer performs; a recycling directive is not one, and
    # requiring it to survive would mean keeping the boilerplate page that
    # otherwise dominates every refusal.
    non_procedural = ("Designates risk", "WEEE", "dismantled", "Directive 20")
    parsed = [
        b.text
        for b in doc.blocks
        if b.type == "callout"
        and b.callout_kind == "warning"
        and not is_boilerplate(b.text)
        and not any(marker in b.text for marker in non_procedural)
    ]
    assert parsed, "parser found no warnings in a manual known to contain them"

    chunks = chunk_document(doc)
    warned = [c for c in chunks if c.has_warning]
    assert warned, "warnings were detected but none survived into a chunk"

    corpus_text = "\n".join(c.text for c in chunks)
    for warning in parsed:
        assert warning in corpus_text, f"warning dropped during chunking: {warning[:80]}"
