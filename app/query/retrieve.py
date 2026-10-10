"""Stage 5 — hybrid retrieval.

Two searches, merged: dense vector search for meaning, lexical
exact-word search for the strings that meaning-based search treats as
noise — part numbers, fault codes, board revisions, terminal labels.
Returns ~50 candidates for the reranker.

Running only the dense half is failure mode 2 (part number matches
nothing by meaning). The lexical half carries an exact-match boost.

Two backends behind one interface. `InMemoryIndex` holds vectors in a
numpy array and scores BM25 in Python: it needs no database, which is
what lets phase 1 be measured against the evaluation set before
Postgres is up, and it is honest for a corpus this size. The Postgres
backend (pgvector HNSW + tsvector GIN) is the same contract and takes
over once the schema is live; retrieval quality measured here should
carry across, because the ranking function is the same.

Fusion is Reciprocal Rank Fusion. RRF combines rankings rather than
scores, which matters because a cosine similarity and a BM25 score are
not the same quantity and normalising them against each other invents a
comparison the numbers do not support.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

RRF_K = 60
"""Rank-fusion damping. The standard value; it flattens the difference
between ranks 1 and 2 enough that one engine cannot dominate on a
single confident hit."""

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs, keeping internal - / . so that part
    codes survive as single tokens: E-412 and 89470K2.GB0 must not
    shatter into digits that match everything."""
    return _TOKEN_RE.findall(text.casefold())


@dataclass
class IndexedChunk:
    chunk_id: str
    document: str
    text: str
    page: int | None = None
    section: str | None = None
    has_warning: bool = False
    # The warning sentences themselves, not just a flag. Tier-3
    # verification has to check that an answer reproduced the warning,
    # and it cannot do that from a boolean.
    warnings: tuple[str, ...] = ()


@dataclass
class Scored:
    chunk: IndexedChunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


@dataclass
class BM25:
    k1: float = 1.5
    b: float = 0.75
    docs: list[list[str]] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    avg_len: float = 0.0

    def fit(self, texts: list[str]) -> BM25:
        self.docs = [tokenize(t) for t in texts]
        self.avg_len = (sum(len(d) for d in self.docs) / len(self.docs)) if self.docs else 0.0
        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))
        return self

    def scores(self, query: str) -> np.ndarray:
        n = len(self.docs)
        out = np.zeros(n, dtype=np.float32)
        if not n:
            return out
        terms = tokenize(query)
        for term in terms:
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.docs):
                tf = doc.count(term)
                if not tf:
                    continue
                norm = 1 - self.b + self.b * (len(doc) / self.avg_len if self.avg_len else 1)
                out[i] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        return out


class InMemoryIndex:
    """Brute-force hybrid retrieval. Exact, no approximation error, and
    fast enough for a corpus of a few thousand chunks — which makes it
    the right thing to measure retrieval quality against before an ANN
    index introduces recall loss of its own."""

    def __init__(self) -> None:
        self.chunks: list[IndexedChunk] = []
        self.vectors: np.ndarray | None = None
        self.bm25 = BM25()

    def add(self, chunks: list[IndexedChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = list(chunks)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.vectors = matrix / np.clip(norms, 1e-12, None)
        self.bm25.fit([c.text for c in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def _ranked(self, scores: np.ndarray, limit: int) -> list[int]:
        if not len(scores):
            return []
        top = np.argsort(-scores)[:limit]
        return [int(i) for i in top if scores[i] > 0]

    def search(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 50,
        documents: set[str] | None = None,
    ) -> list[Scored]:
        if self.vectors is None or not self.chunks:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        dense = self.vectors @ q
        lexical = self.bm25.scores(query)

        if documents is not None:
            mask = np.array([c.document in documents for c in self.chunks])
            dense = np.where(mask, dense, -np.inf)
            lexical = np.where(mask, lexical, 0.0)

        pool = max(limit * 4, 200)
        dense_order = [int(i) for i in np.argsort(-dense)[:pool] if np.isfinite(dense[i])]
        lexical_order = self._ranked(lexical, pool)

        dense_rank = {idx: r for r, idx in enumerate(dense_order, start=1)}
        lexical_rank = {idx: r for r, idx in enumerate(lexical_order, start=1)}

        fused: dict[int, float] = {}
        for idx, rank in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for idx, rank in lexical_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        best = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
        return [
            Scored(
                chunk=self.chunks[idx],
                score=score,
                dense_rank=dense_rank.get(idx),
                lexical_rank=lexical_rank.get(idx),
            )
            for idx, score in best
        ]
