"""Transcriber protocol. Phase 6.

Push-to-talk, not live streaming: site audio is noisy and streaming buys
little here. Returns a transcript plus a confidence score, and the
transcript is always shown back to the user for confirmation before it is
acted on — a wrong filler word costs nothing, a wrong part number costs
an hour (failure modes 12 and 13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float
    model: str


@runtime_checkable
class Transcriber(Protocol):
    name: str

    async def transcribe(self, audio: bytes, mime_type: str) -> Transcript: ...
