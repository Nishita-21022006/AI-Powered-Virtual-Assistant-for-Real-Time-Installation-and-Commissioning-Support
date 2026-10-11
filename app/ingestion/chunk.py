"""Chunking — by section structure, never by fixed word count.

This is the part that decides answer quality. Chunking badly is the most
common way to build a RAG system that retrieves correctly and still gives
dangerous answers: a fixed-width split puts "Step 4: tighten the mounting
screws" in one chunk and "WARNING: maximum screw head height is 3 mm" in
the next, and a correct retrieval then returns a confident, complete,
silently lethal answer.

HARD RULE: a safety warning is never separated from the procedure it
governs. If a chunk contains a step, it contains that step's warnings.

How the rule is enforced here
-----------------------------
Sections are the unit. A section that fits stays whole, so its warnings
travel with its steps for free. The rule only comes under pressure when
a section is too large for one chunk, and the honest answer there is
that no split point is provably safe: a warning can govern steps before
it, after it, or the whole procedure, and the document does not say
which.

So a split section does not try to guess. Every chunk derived from a
section that carries warnings gets ALL of that section's warnings
attached. The duplication costs a little index space and makes the
guarantee unconditional rather than dependent on correctly inferring
scope. Given the cost of being wrong once, that is the right trade.

Warnings arrive as `callout` blocks from the parser, detected by the
pictogram drawn beside them — in this corpus a warning carries no
"WARNING:" string, so text matching finds nothing. See parse.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.parse import Block, ParsedDocument

# Chars, not tokens: the exact budget does not matter much, and a char
# proxy avoids loading a tokeniser during ingestion. ~4 chars/token.
MAX_CHUNK_CHARS = 4800
MIN_CHUNK_CHARS = 200

_STEP_RE = re.compile(
    r"^\s*(?:step\s+\d+|[1-9]\d?[.)]\s|[-•]\s)", re.IGNORECASE
)
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:connect|mount|fit|tighten|remove|insert|set|press|switch|screw|"
    r"attach|align|close|open|check|verify|install|wire|plug|unplug)\b",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    content_type: str
    section: str | None = None
    section_number: str | None = None
    has_warning: bool = False
    warnings: tuple[str, ...] = ()
    ordinal: int = 0
    char_start: int = 0
    char_end: int = 0
    table_refs: tuple[str, ...] = ()
    figure_refs: tuple[str, ...] = ()


@dataclass
class _Section:
    heading: Block | None
    blocks: list[Block] = field(default_factory=list)

    @property
    def title(self) -> str | None:
        return self.heading.text if self.heading else None

    @property
    def number(self) -> str | None:
        return self.heading.section_number if self.heading else None

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            b.text for b in self.blocks
            if b.type == "callout" and b.callout_kind == "warning"
        )


# Legal and boilerplate front matter: copyright, trademark, disposal
# and symbol-legend pages. Every manual opens with a near-identical
# copy, so these chunks are both unanswerable and highly similar to each
# other — which makes them the default "closest match" for any question
# the corpus does not cover, and turns an honest refusal into a wall of
# copyright notices. Dropped at ingestion rather than filtered at query
# time: nothing downstream can answer a question from them.
_BOILERPLATE_MARKERS = (
    "intended purpose",
    "subject to the copyright",
    "registered and unregistered trademarks",
    "qualified personnel",
    "provided without warranty",
    "urhg",
    "weee",
    "dismantling",
    "demontage",
    "technical changes reserved",
)
_BOILERPLATE_THRESHOLD = 3


def is_boilerplate(text: str) -> bool:
    lowered = text.casefold()
    return sum(m in lowered for m in _BOILERPLATE_MARKERS) >= _BOILERPLATE_THRESHOLD


def _is_step(block: Block) -> bool:
    if block.type != "paragraph":
        return False
    return bool(_STEP_RE.match(block.text) or _IMPERATIVE_RE.match(block.text))


# A numbered heading that the parser left as a paragraph: "7.
# Specifications", "4.2 Commissioning". Short, starts with a section
# number, no sentence-ending punctuation.
_MISSED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.!?]{2,60})$")


def _recover_headings(blocks: list[Block]) -> list[Block]:
    """Promote heading-looking paragraphs to headings.

    Structure-aware chunking is only as good as the structure the parser
    can see, and it detects headings by font weight and size. One manual
    in this corpus is a draft with inconsistent formatting, so real
    headings come through as ordinary paragraphs — and a section's
    content then runs on into the next one.

    The cost is concrete: a technical-specifications table ended up
    inside a chunk titled "Separate cancel push-button for bed", and a
    reranker reading that chunk against "what temperature range" quite
    reasonably scored it as off-topic. The passage was retrieved at
    rank 1 and dropped before it reached the model.

    Recovering the numbering is a second, weaker signal for the same
    thing. Conservative on purpose: a numbered line is only promoted
    when it is short and reads like a title, so a step like
    "3. Tighten the four screws to 2.5 Nm." is left alone.
    """
    out: list[Block] = []
    for block in blocks:
        match = (
            _MISSED_HEADING_RE.match(block.text.strip())
            if block.type == "paragraph"
            else None
        )
        if match and not _is_step(block):
            out.append(
                Block(
                    type="heading",
                    text=match.group(2).strip(),
                    page=block.page,
                    bbox=block.bbox,
                    level=match.group(1).count(".") + 1,
                    section_number=match.group(1),
                )
            )
        else:
            out.append(block)
    return out


def _sections(doc: ParsedDocument) -> list[_Section]:
    sections: list[_Section] = []
    current = _Section(heading=None)
    for block in _recover_headings(doc.blocks):
        if block.type == "heading":
            if current.blocks or current.heading:
                sections.append(current)
            current = _Section(heading=block)
        else:
            current.blocks.append(block)
    if current.blocks or current.heading:
        sections.append(current)
    return [s for s in sections if s.blocks]


def _content_type(blocks: list[Block]) -> str:
    if any(b.type == "table" for b in blocks):
        return "table"
    if any(b.type == "figure" for b in blocks):
        return "figure_caption"
    if sum(_is_step(b) for b in blocks) >= 2:
        return "procedure"
    return "prose"


def _render_table(block: Block, tables: dict[str, object]) -> str:
    """Tables go into the index three ways, because no single rendering
    serves every query.

    The spec requires indexing the table and a prose description of it;
    the plain row text is the third, and it is the one that makes a part
    number findable — markdown pipes and prose scaffolding both sit
    between the query term and the match for lexical search.

    Flattening a table to one of these alone is failure mode 7: the
    row-to-column association is where the answer lives.
    """
    table = tables.get(block.ref or "")
    if table is None:
        return block.text
    parts = [table.to_markdown(), table.to_text()]
    prose = table.to_prose()
    if prose:
        parts.append(prose)
    return "\n".join(p for p in parts if p.strip())


def _render(section: _Section, blocks: list[Block], tables: dict[str, object]) -> str:
    parts: list[str] = []
    if section.heading:
        number = section.number
        parts.append(f"{number} {section.title}" if number else section.title or "")
    for b in blocks:
        text = _render_table(b, tables) if b.type == "table" else b.text
        if text.strip():
            parts.append(text)
    return "\n".join(p for p in parts if p.strip())


def _groups(section: _Section) -> list[list[Block]]:
    """Split an oversized section at block boundaries.

    Never splits mid-block, and never leaves a runt: a trailing group
    below the minimum is merged back into the previous one.
    """
    total = sum(len(b.text) for b in section.blocks)
    if total <= MAX_CHUNK_CHARS:
        return [section.blocks]

    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for block in section.blocks:
        if current and size + len(block.text) > MAX_CHUNK_CHARS:
            groups.append(current)
            current, size = [], 0
        current.append(block)
        size += len(block.text)
    if current:
        groups.append(current)

    if len(groups) > 1 and sum(len(b.text) for b in groups[-1]) < MIN_CHUNK_CHARS:
        groups[-2].extend(groups.pop())
    return groups


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = 0
    tables = {t.id: t for t in doc.tables}

    for section in _sections(doc):
        section_warnings = section.warnings
        groups = _groups(section)
        split = len(groups) > 1

        for group in groups:
            text = _render(section, group, tables)
            if not text.strip() or is_boilerplate(text):
                continue

            pages = [b.page for b in group] or [0]
            content_type = _content_type(group)

            # The rule. An unsplit section already holds its warnings in
            # its own text. A split one re-attaches all of them to every
            # piece that carries procedural content, because which steps
            # a given warning governs is not recoverable from the source.
            carried = section_warnings
            if split and section_warnings:
                has_steps = any(_is_step(b) for b in group)
                already = any(
                    b.type == "callout" and b.callout_kind == "warning" for b in group
                )
                if has_steps and not already:
                    text = text + "\n" + "\n".join(section_warnings)

            chunks.append(
                Chunk(
                    text=text,
                    page_start=min(pages),
                    page_end=max(pages),
                    content_type=content_type,
                    section=section.title,
                    section_number=section.number,
                    has_warning=bool(carried),
                    warnings=carried,
                    ordinal=len(chunks),
                    char_start=cursor,
                    char_end=cursor + len(text),
                    table_refs=tuple(b.ref for b in group if b.type == "table" and b.ref),
                    figure_refs=tuple(b.ref for b in group if b.type == "figure" and b.ref),
                )
            )
            cursor += len(text)

    return chunks
