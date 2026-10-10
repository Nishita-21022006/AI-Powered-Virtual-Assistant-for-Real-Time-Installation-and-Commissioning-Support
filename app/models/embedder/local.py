"""Embedder — BGE-M3, local, CPU, free, ~2 GB.

Dense and sparse retrieval in one model, which is exactly the hybrid
search this system needs without running two. Multilingual, which
matters for a corpus with untranslated German sections.

Downloaded from Hugging Face: no API key, no rate limit, no daily cap.
Paying an API to run a model this size would be spending money to add a
dependency.

Only the dense side is used here. BGE-M3's learned sparse vectors need
FlagEmbedding and its own index; the architecture already specifies a
Postgres keyword index for the lexical half of hybrid retrieval, so the
exact-string path goes there instead of through a second vector space.

The model loads once and is cached on the instance: on CPU, loading
costs seconds and encoding a corpus costs minutes, so a per-call load
would dominate everything.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import cached_property

DEFAULT_MODEL = "BAAI/bge-m3"
DIM = 1024

# BGE-M3 is trained so that a query and a passage embed into the same
# space without an instruction prefix, unlike bge-large-en. Kept as a
# named constant so a model swap that *does* need one has an obvious
# place to put it.
QUERY_PREFIX = ""


class LocalEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = None):
        self.name = model
        self.dim = DIM
        self._cache_dir = cache_dir or os.environ.get("MODEL_CACHE_DIR") or None

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.name, cache_folder=self._cache_dir, device="cpu"
        )

    def _encode(self, texts: Sequence[str], batch_size: int) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    async def embed_documents(
        self, texts: Sequence[str], batch_size: int = 8
    ) -> list[list[float]]:
        return self._encode(texts, batch_size)

    async def embed_query(self, text: str) -> list[float]:
        return self._encode([QUERY_PREFIX + text], batch_size=1)[0]

    def embed_documents_sync(
        self, texts: Sequence[str], batch_size: int = 8
    ) -> list[list[float]]:
        """Ingestion runs in a worker process, not an event loop."""
        return self._encode(texts, batch_size)

    def embed_query_sync(self, text: str) -> list[float]:
        return self._encode([QUERY_PREFIX + text], batch_size=1)[0]
