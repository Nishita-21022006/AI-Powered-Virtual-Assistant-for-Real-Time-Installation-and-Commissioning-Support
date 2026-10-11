"""Stage 9 — output verification. Guardrail tier 3, the safety net.

Deterministic string checks, not AI. An answer is rejected if:
  - it cites a document that was not in the retrieved set
  - any number, tolerance or part code in it is absent from the sources
  - it answers a procedure without carrying that procedure's warnings
  - it carries no citation at all (never shown, under any circumstance)

These checks are cheap, testable, and cannot themselves hallucinate.
They catch the most dangerous failure: a confident invented value.

Everything here is a pure function over strings. No I/O, no model call,
no settings lookup, so this layer cannot fail open because a dependency
is down: the only route to an accept is passing the checks.

MATCHING TRADEOFF (the decision this module exists to get right)
----------------------------------------------------------------
"Appears verbatim" cannot mean byte equality -- a generator writes
"3 mm" where the manual prints "3mm", and "2.5 Nm" where a European
manual prints "2,5 Nm". A byte-equal check would reject correct answers
all day, which teaches integrators to ignore the guardrail. So the
answer and the cited sources are run through the *same* canonicaliser and
the canonical forms compared exactly. Normalisation applied:

  - number/unit spacing removed: "3 mm" == "3mm"
  - unit aliases folded: "Nm" == "N.m" == "N*m"; "ohms" == "Ohm"
  - decimal comma honoured for 1-2 trailing digits: "2,5" == "2.5"
  - trailing fractional zeros and leading zeros dropped: "2.50" == "2.5",
    "03" == "3"
  - thousands separators: comma + exactly three digits reads as a
    thousands group ("1,000" == "1000"); point + three digits reads as a
    decimal ("3.000" == "3"). English-locale corpus, so that is the
    common reading. The residual ambiguity is real but harmless in both
    directions: whichever reading is taken, it is taken identically on
    the answer side and the source side, so a genuine match survives and
    a genuine mismatch still rejects.
  - identifiers fold on case and on separators: "E-412" == "E412",
    "TB-3" == "TB3". This deliberately collapses a distinction a vendor
    could in principle use ("A-1" vs "A1"); the alternative is rejecting
    answers that merely hyphenate differently from the manual.

Two places where the check is deliberately *looser* than it could be,
both to avoid false rejections, and both documented because they are
where an invented value could in theory slip past:

  - a figure's unit must appear somewhere in the cited sources, but not
    adjacent to that figure. Manuals put the unit in a table header and
    the figure in a cell, so requiring adjacency produces false
    rejections. The bare-number check is the load-bearing one.
  - single-letter and English-word units ("a", "in", "m") are credited
    only when attached to a number, never from standalone prose, because
    "in" and "a" occur in every paragraph of English.

Everywhere else the bias runs the other way: an unrecognised bracketed
span counts as a citation claim and therefore as an invented citation; a
spelled-out number ("two screws" against a source saying "2") does not
match and rejects. A false rejection costs an answer the user could have
had. A false accept can injure someone on a ladder.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

WARNING_COVERAGE_FLOOR = 0.6
"""Fraction of a warning's significant words that must survive into the
answer for the warning to count as carried. Every number and identifier
in the warning is required unconditionally, regardless of this."""


@dataclass(frozen=True)
class SourcePassage:
    """A retrieved chunk, as it will be cited back to the user."""

    doc_id: str
    text: str
    section: str | None = None
    page: int | None = None
    warnings: tuple[str, ...] = ()


class Rejection(StrEnum):
    NO_CITATION = "no_citation"
    CITATION_NOT_RETRIEVED = "citation_not_in_retrieved_set"
    UNSUPPORTED_VALUE = "value_absent_from_cited_sources"
    OMITTED_WARNING = "warning_omitted_from_answer"


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reasons: tuple[Rejection, ...] = ()
    details: tuple[str, ...] = ()
    cited_doc_ids: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.accepted


_MARKER_RE = re.compile(r"\[([^\[\]]{1,200})\]")

# Identifier patterns run before the numeric scan and their matches are
# masked out, so a part code never decomposes into a bare number that some
# other source happens to contain: "E-412" must not be read as 412.
_IDENTIFIER_RES = (
    re.compile(r"\b[A-Za-z]{1,10}[-/][A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*\b"),
    re.compile(r"\b[A-Za-z]{1,10}\d[A-Za-z0-9]*\b"),
    re.compile(r"\b\d+(?:-\d+)+\b"),
)

_UNIT = (
    r"(?:%|°\s?[CFcf]|"
    r"[A-Za-zµμΩ][A-Za-zµμΩ·.²³/]*)"
)
_NUMBER = r"(?<![A-Za-z0-9.,])(?:(?<![^\s(])-)?\d+(?:[.,]\d+)*"
_NUM_UNIT_RE = re.compile(rf"({_NUMBER})[\s ]{{0,2}}({_UNIT})?")
_WORD_RE = re.compile(
    r"[A-Za-zµμΩ°%][A-Za-zµμΩ·.²³/%°]*"
)

_UNIT_ALIASES = {
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm",
    "millimetres": "mm", "cm": "cm", "m": "m", "um": "um", "mm2": "mm2", "cm2": "cm2",
    "in": "in", "inch": "in", "inches": "in", "ft": "ft", "feet": "ft", "mil": "mil",
    # Nm is newton-metre in an installation manual; nanometres do not occur
    # in one, so folding the case-insensitive collision is safe.
    "nm": "n*m", "nmm": "n*mm", "lbin": "lb*in", "lbft": "lb*ft", "ozin": "oz*in",
    "v": "v", "volt": "v", "volts": "v", "vac": "vac", "vdc": "vdc", "kv": "kv", "mv": "mv",
    "a": "a", "amp": "a", "amps": "a", "ampere": "a", "amperes": "a", "ma": "ma", "ka": "ka",
    "w": "w", "kw": "kw", "mw": "mw", "va": "va", "kva": "kva",
    "ohm": "ohm", "ohms": "ohm", "kohm": "kohm", "mohm": "mohm",
    "hz": "hz", "khz": "khz", "mhz": "mhz",
    "%": "%", "pct": "%", "percent": "%",
    "degc": "degc", "degf": "degf", "k": "k",
    "bar": "bar", "psi": "psi", "pa": "pa", "kpa": "kpa", "mpa": "mpa",
    "kg": "kg", "g": "g", "lb": "lb", "lbs": "lb", "n": "n", "kn": "kn",
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "ms": "ms", "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "rpm": "rpm", "awg": "awg", "ip": "ip",
}

# Credited as a unit only when written against a number. Standalone these
# are ordinary English words and would make the unit check vacuous.
_AMBIGUOUS_STANDALONE_UNITS = frozenset(
    {"a", "in", "m", "s", "g", "n", "w", "v", "h", "k", "min", "ms", "ft"}
)

_WARNING_LABEL_RE = re.compile(
    r"^\s*(?:!+\s*)?(?:WARNING|CAUTION|DANGER|NOTICE|ATTENTION|IMPORTANT)\b[\s:.!-]*",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "been", "before", "by", "can", "do",
        "does", "for", "from", "has", "have", "if", "in", "into", "is", "it", "its",
        "may", "must", "not", "of", "on", "only", "or", "should", "that", "the", "then",
        "there", "these", "this", "to", "under", "until", "when", "which", "while",
        "will", "with", "without", "you", "your",
    )
)


def _fold(text: str) -> str:
    return " ".join(text.casefold().split())


def _canon_number(raw: str) -> str:
    negative = raw.startswith("-")
    body = raw.lstrip("-")
    cut = max(body.rfind("."), body.rfind(","))
    if cut == -1:
        whole, frac = body, ""
    else:
        head, tail = body[:cut], body[cut + 1 :]
        head_digits = head.replace(".", "").replace(",", "")
        thousands_group = len(tail) == 3 and (
            body[cut] == "," or "." in head or "," in head
        )
        if thousands_group:
            whole, frac = head_digits + tail, ""
        else:
            whole, frac = head_digits, tail
    whole = whole.lstrip("0") or "0"
    frac = frac.rstrip("0")
    value = whole + "." + frac if frac else whole
    return "-" + value if negative and value != "0" else value


def _canon_unit(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = (
        raw.strip()
        .casefold()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("Ω", "ohm")
        .replace("²", "2")
        .replace("³", "3")
        .replace("·", "")
        .replace(".", "")
        .replace("-", "")
        .replace(" ", "")
    )
    if key.startswith("°"):
        key = "deg" + key[1:]
    return _UNIT_ALIASES.get(key)


def _canon_identifier(raw: str) -> str:
    return re.sub(r"[-/._\s]", "", raw).upper()


@dataclass(frozen=True)
class _Figure:
    number: str
    unit: str | None
    raw: str


@dataclass(frozen=True)
class _Identifier:
    canonical: str
    raw: str


_LIST_ORDINAL_RE = re.compile(r"^[ \t]*(?:\d{1,2}[.)]|[-•*])[ \t]+", re.MULTILINE)

# A document name or page reference is a pointer to a source, not a
# claim about the equipment. Outside brackets they were scanned as
# content, so an answer mentioning "89470J6.GB0.pdf p27" in prose was
# rejected for stating figures absent from its sources — the source
# being the very document it was naming.
#
# The page pattern is deliberately lowercase-only: `p27` is a page,
# `P12` is a parameter name in this corpus and must still be checked.
_DOC_REF_RE = re.compile(
    r"\b[\w.-]+\.pdf\b"
    r"|\b\d{5}[A-Z]\d[A-Z0-9.]*\b"
    r"|\bp\.?\s?\d{1,3}\b"
)


def _strip_document_refs(text: str) -> str:
    return _DOC_REF_RE.sub(" ", text)


def _strip_list_ordinals(text: str) -> str:
    """Remove "1. " / "2) " / "- " at the start of a line.

    A step number is structure, not a claim. Left in place, a numbered
    procedure makes the verifier demand that the figures 1, 2, 3 appear
    verbatim in a cited source, and a perfectly good five-step answer is
    rejected for inventing the number 4. Only leading, line-anchored
    ordinals go: "4 mm" mid-sentence is a real measurement and must
    still be checked.
    """
    return _LIST_ORDINAL_RE.sub("", text)


def _scan(text: str) -> tuple[tuple[_Figure, ...], tuple[_Identifier, ...], frozenset[str]]:
    """Pull every figure, identifier and unit out of a blob of text."""
    identifiers: list[_Identifier] = []
    masked = text
    for pattern in _IDENTIFIER_RES:
        out: list[str] = []
        last = 0
        for match in pattern.finditer(masked):
            raw = match.group(0)
            mixed = any(c.isdigit() for c in raw) and any(c.isalpha() for c in raw)
            if not mixed and not re.fullmatch(r"\d+(?:-\d+)+", raw):
                continue
            if _canon_unit(raw) is not None:
                continue
            identifiers.append(_Identifier(_canon_identifier(raw), raw))
            out.append(masked[last : match.start()])
            out.append(" " * len(raw))
            last = match.end()
        out.append(masked[last:])
        masked = "".join(out)

    figures: list[_Figure] = []
    units: set[str] = set()
    for match in _NUM_UNIT_RE.finditer(masked):
        unit = _canon_unit(match.group(2))
        figures.append(_Figure(_canon_number(match.group(1)), unit, match.group(0).strip()))
        if unit is not None:
            units.add(unit)

    for match in _WORD_RE.finditer(text):
        unit = _canon_unit(match.group(0))
        if unit is not None and match.group(0).casefold() not in _AMBIGUOUS_STANDALONE_UNITS:
            units.add(unit)

    return tuple(figures), tuple(identifiers), frozenset(units)


def _citations(
    answer: str, retrieved: Sequence[SourcePassage], declared: Sequence[str] | None
) -> tuple[set[str], list[str]]:
    known = {p.doc_id: _fold(p.doc_id) for p in retrieved}
    cited: set[str] = set()
    unknown: list[str] = []

    def resolve(claim: str) -> None:
        folded = _fold(claim)
        if not folded:
            return
        hits = [doc for doc, folded_doc in known.items() if folded_doc and folded_doc in folded]
        if hits:
            cited.update(hits)
        else:
            unknown.append(claim.strip())

    for marker in _MARKER_RE.finditer(answer):
        for part in marker.group(1).split(";"):
            if part.strip():
                resolve(part)
    for claim in declared or ():
        resolve(claim)
    return cited, unknown


def _significant_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[A-Za-z]{3,}", text.casefold()) if word not in _STOPWORDS
    }


def verify(
    answer: str,
    retrieved: Sequence[SourcePassage],
    cited_doc_ids: Sequence[str] | None = None,
) -> Verdict:
    """Tier 3. Accept the answer, or reject it with every reason it failed.

    `cited_doc_ids` is the generator's structured citation list where it
    produces one; markers written into the answer text are always parsed
    too, and the two are unioned. Any bracketed span that does not resolve
    to a retrieved document is an invented citation -- the tier-2 prompt
    reserves brackets for citations, so there is no benign reading of one.

    A warning carried by any cited passage must be carried by the answer.
    This module does not try to decide whether the answer "is procedural"
    before demanding it: that is exactly the model judgment a deterministic
    tier is here to avoid, and the spec already requires a warning to
    travel with the text it governs.
    """
    reasons: list[Rejection] = []
    details: list[str] = []

    cited, unknown = _citations(answer, retrieved, cited_doc_ids)
    if unknown:
        reasons.append(Rejection.CITATION_NOT_RETRIEVED)
        details += ["citation not in retrieved set: " + repr(c) for c in sorted(set(unknown))]

    if not cited:
        reasons.insert(0, Rejection.NO_CITATION)
        details.insert(0, "answer carries no citation resolving to a retrieved document")
        return Verdict(False, tuple(reasons), tuple(details))

    sources = [p for p in retrieved if p.doc_id in cited]
    # A chunk's warnings are part of its text as far as support goes: a figure
    # printed only inside the WARNING is still printed in the cited source.
    source_blob = "\n".join("\n".join((p.text,) + tuple(p.warnings)) for p in sources)
    source_figures, source_identifiers, source_units = _scan(source_blob)
    source_numbers = {f.number for f in source_figures}
    source_ids = {i.canonical for i in source_identifiers}

    body = _strip_document_refs(_strip_list_ordinals(_MARKER_RE.sub(" ", answer)))
    answer_figures, answer_identifiers, _ = _scan(body)
    answer_numbers = {f.number for f in answer_figures}
    answer_ids = {i.canonical for i in answer_identifiers}

    unsupported: list[str] = []
    for figure in answer_figures:
        if figure.number not in source_numbers:
            unsupported.append("figure " + repr(figure.raw) + " absent from cited sources")
        elif figure.unit is not None and figure.unit not in source_units:
            unsupported.append("unit of " + repr(figure.raw) + " absent from cited sources")
    for identifier in answer_identifiers:
        if identifier.canonical not in source_ids:
            unsupported.append("identifier " + repr(identifier.raw) + " absent from cited sources")
    if unsupported:
        reasons.append(Rejection.UNSUPPORTED_VALUE)
        details += sorted(set(unsupported))

    folded_answer = _fold(body)
    missing: list[str] = []
    for passage in sources:
        for warning in passage.warnings:
            text = _WARNING_LABEL_RE.sub("", warning).strip()
            figures, identifiers, _ = _scan(text)
            if any(f.number not in answer_numbers for f in figures) or any(
                i.canonical not in answer_ids for i in identifiers
            ):
                missing.append(passage.doc_id + ": " + repr(warning))
                continue
            words = _significant_words(text)
            if not words:
                if _fold(text) not in folded_answer:
                    missing.append(passage.doc_id + ": " + repr(warning))
                continue
            carried = sum(1 for word in words if word in folded_answer) / len(words)
            if carried < WARNING_COVERAGE_FLOOR:
                missing.append(passage.doc_id + ": " + repr(warning))
    if missing:
        reasons.append(Rejection.OMITTED_WARNING)
        details += ["warning not carried into answer -- " + w for w in sorted(set(missing))]

    return Verdict(
        accepted=not reasons,
        reasons=tuple(reasons),
        details=tuple(details),
        cited_doc_ids=tuple(sorted(cited)),
    )
