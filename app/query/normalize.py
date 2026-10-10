"""Stage 1 — input normalization. Guardrail tier 0.

Trim whitespace, reject empty or absurd input. Nonsense never reaches
retrieval, and voice transcripts are shown back to the user for
confirmation before being acted on (phase 6).

Deliberately thin. This stage exists to catch input that is not a
question at all, not to judge whether a question is answerable — that
is the score gate's job at stage 7, and it decides with evidence rather
than with a heuristic about the wording. Rejecting too eagerly here
would turn an honest "I don't have documentation on that, here are the
closest sections" into a blunt refusal that helps nobody.
"""

from __future__ import annotations

import re
import unicodedata

MIN_CHARS = 3
MAX_CHARS = 2000

_WHITESPACE_RE = re.compile(r"\s+")
_HAS_LETTER_OR_DIGIT = re.compile(r"[^\W_]", re.UNICODE)


class NormalizationError(ValueError):
    """Input rejected at tier 0. The message is shown to the user."""


def normalize(text: str) -> str:
    if text is None:
        raise NormalizationError("no question was provided")

    # NFKC so that full-width and compatibility forms of part numbers
    # match the ASCII in the manuals.
    cleaned = _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()

    if not cleaned:
        raise NormalizationError("no question was provided")
    if len(cleaned) < MIN_CHARS:
        raise NormalizationError("that is too short to search on — please say more")
    if len(cleaned) > MAX_CHARS:
        raise NormalizationError(
            f"that question is {len(cleaned)} characters; please shorten it to "
            f"under {MAX_CHARS}"
        )
    if not _HAS_LETTER_OR_DIGIT.search(cleaned):
        raise NormalizationError("that contains no searchable text")

    return cleaned
