"""Generator protocol.

Takes a question and up to five passages; returns an answer with
citations, or an explicit abstention. Streaming is part of the contract —
first words on screen inside the latency budget matter more than total
completion time.

All implementations speak an OpenAI-compatible interface. That is what
makes the Ollama (development) to vLLM (production) to hosted-API moves
config changes rather than rewrites.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    doc_id: str
    text: str
    section: str | None = None
    page: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Generation:
    text: str
    cited_doc_ids: tuple[str, ...]
    model: str
    abstained: bool = False


@runtime_checkable
class Generator(Protocol):
    name: str

    async def generate(
        self, question: str, passages: Sequence[Passage]
    ) -> Generation: ...

    def stream(
        self, question: str, passages: Sequence[Passage]
    ) -> AsyncIterator[str]: ...
