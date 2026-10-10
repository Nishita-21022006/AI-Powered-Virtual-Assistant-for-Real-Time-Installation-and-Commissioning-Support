"""Embedder protocol.

Embeds chunks at ingestion time and queries at retrieval time, and must
report which model produced a given vector — switching embedding model
invalidates the whole index, so the index records its model and blocks
queries against a mixed-model state (failure mode 17).

The most expensive decision to reverse in the system: cloud and local
embedding models produce incompatible vectors. Extracted text and chunk
boundaries are therefore stored separately from vectors from day one, so
a model switch is a batch re-embed, not a re-parse.

`name` and `dim` are part of the contract, not conveniences: every vector
is written tagged with the name, and dim must match the column width the
index was built for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
