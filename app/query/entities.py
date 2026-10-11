"""Which product, bus or terminal is this passage about?

Every genuine wrong answer in the phase 3 run was the right answer
about the wrong thing: the SEM-IP temperature given for the SEM, the
ZBUS device limit for the R-ABUS, the ward supply current for the bed
terminals. The figures were real and correctly cited, so neither
retrieval scoring nor tier-3 verification had any reason to object.

What makes these confusable is that they are near-synonyms to an
embedding model and near-identical as strings. "SEM" is a substring of
"SEM-IP", so naive matching reads a passage about one as a passage
about the other — which is the error itself, in miniature.

So entities are grouped into FAMILIES of mutually exclusive
alternatives. Within a family, naming one is implicitly ruling out the
others, and a passage about a sibling is evidence AGAINST relevance
rather than merely neutral. Across families there is no such relation:
a passage can be about the SEM and the DBUS at once.

Names come from counting the corpus, not from guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Longest names first within each family: "SEM-IP" must be matched
# before "SEM", or every SEM-IP passage reads as a SEM passage.
FAMILIES: dict[str, tuple[str, ...]] = {
    "bus": ("R-ABUS", "DBUS", "ZBUS", "ABUS", "BBUS"),
    "room_module": ("SEM-IP", "SEM+", "SEM"),
    "unit": ("SCVIP", "SCIP+", "SCIP", "STIP", "SINF", "SCL", "SRT", "SRD", "SBU"),
}

# Spoken and written forms that mean the same device. An installer asks
# about "the touch terminal", not "the STIP".
ALIASES: dict[str, tuple[str, ...]] = {
    "STIP": ("systevo touch", "touch terminal", "touch ip"),
    "SCVIP": ("care view", "systevo care view"),
    "SEM": ("systevo electronic module", "electronic module"),
    "SEM-IP": ("systevo electronic module ip", "electronic module ip"),
    "SCL": ("corridor lamp", "systevo corridor lamp"),
    "SINF": ("information display", "systevo information"),
    "SCIP+": ("systevo control ip", "control ip"),
}

_MEMBER_TO_FAMILY = {
    member: family for family, members in FAMILIES.items() for member in members
}


def _pattern(name: str) -> re.Pattern[str]:
    """Word-boundary match that survives + and -.

    `\\b` does not work after a '+', so SEM+ needs its own handling or
    "SEM+" matches inside nothing and "SEM" matches inside "SEM+".
    """
    escaped = re.escape(name)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern[str]] = {
    member: _pattern(member) for member in _MEMBER_TO_FAMILY
}
_ALIAS_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    member: tuple(re.compile(re.escape(a), re.IGNORECASE) for a in aliases)
    for member, aliases in ALIASES.items()
}


@dataclass(frozen=True)
class Mentions:
    """Which family members a piece of text refers to."""

    by_family: dict[str, set[str]]

    def members(self) -> set[str]:
        return {m for ms in self.by_family.values() for m in ms}

    def __bool__(self) -> bool:
        return bool(self.by_family)


def detect(text: str) -> Mentions:
    found: dict[str, set[str]] = {}
    for family, members in FAMILIES.items():
        hit: set[str] = set()
        consumed = text
        for member in members:  # longest first
            if _PATTERNS[member].search(consumed):
                hit.add(member)
                # Remove the match so a shorter sibling cannot also claim
                # it: without this every "SEM-IP" also counts as "SEM".
                consumed = _PATTERNS[member].sub(" ", consumed)
            else:
                for alias in _ALIAS_PATTERNS.get(member, ()):
                    if alias.search(consumed):
                        hit.add(member)
                        consumed = alias.sub(" ", consumed)
                        break
        if hit:
            found[family] = hit
    return Mentions(found)


def agreement(question: Mentions, passage: Mentions) -> float:
    """How well a passage's subject matches the question's, in [-1, 1].

    Positive when the passage is about what was asked. Negative when it
    is about a sibling instead — the SEM-IP page for a SEM question.
    Zero when the question names nothing to check against, which is the
    common case and must stay neutral rather than penalising everything.
    """
    if not question:
        return 0.0

    score = 0.0
    families = 0
    for family, wanted in question.by_family.items():
        families += 1
        present = passage.by_family.get(family, set())
        if not present:
            continue  # silent on this family: neither support nor conflict
        if wanted & present:
            score += 1.0
        else:
            # Names a sibling and not the one asked about. This is the
            # signal that was missing: evidence against, not absence of
            # evidence.
            score -= 1.0
    return score / families if families else 0.0
