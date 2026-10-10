"""Evaluation runner — executes the question set against the current
configuration and writes a comparable report. Run on every change.

Phase 1 measures retrieval alone: no generation, no interface. A
question counts as answered correctly only if a retrieved chunk comes
from the expected document AND contains the question's verbatim
supporting quote, so retrieving the right manual at the wrong page
scores nothing.

The parsed-and-embedded corpus is cached to disk. Embedding ~8 manuals
on CPU takes minutes; re-running the eval after a retrieval change
should take seconds, or the eval stops being something anyone runs on
every change.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from app.eval.dataset import EvalSet, load
from app.eval.metrics import RetrievalReport, RetrievalResult, RetrievedChunk, rank_of_hit
from app.ingestion.chunk import chunk_document
from app.ingestion.parse import ParseError, parse_pdf
from app.models.embedder.local import LocalEmbedder
from app.query.retrieve import IndexedChunk, InMemoryIndex

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "Documents"
CACHE = ROOT / ".cache" / "corpus_index.pkl"

CATALOG = "Catalog_EN_11052024_fromWEB.pdf"


@dataclass
class BuiltCorpus:
    chunks: list[IndexedChunk]
    vectors: list[list[float]]
    model: str
    stats: dict


def build_corpus(
    include_catalog: bool = False, embedder: LocalEmbedder | None = None
) -> BuiltCorpus:
    embedder = embedder or LocalEmbedder(cache_dir=str(ROOT / "models" / "weights"))
    paths = sorted(p for p in CORPUS_DIR.glob("*.pdf"))
    if not include_catalog:
        paths = [p for p in paths if p.name != CATALOG]

    chunks: list[IndexedChunk] = []
    stats: dict = {"documents": {}, "failed": {}}

    for path in paths:
        started = time.time()
        try:
            doc = parse_pdf(path, extract_figure_images=False)
        except ParseError as exc:
            stats["failed"][path.name] = str(exc)
            continue
        doc_chunks = chunk_document(doc)
        for c in doc_chunks:
            chunks.append(
                IndexedChunk(
                    chunk_id=f"{path.name}#{c.ordinal}",
                    document=path.name,
                    text=c.text,
                    page=c.page_start,
                    section=c.section,
                    has_warning=c.has_warning,
                    warnings=tuple(c.warnings),
                )
            )
        stats["documents"][path.name] = {
            "pages": doc.page_count,
            "chunks": len(doc_chunks),
            "warned_chunks": sum(c.has_warning for c in doc_chunks),
            "parse_seconds": round(time.time() - started, 1),
        }

    vectors = embedder.embed_documents_sync([c.text for c in chunks], batch_size=8)
    return BuiltCorpus(chunks=chunks, vectors=vectors, model=embedder.name, stats=stats)


def save_corpus(corpus: BuiltCorpus, path: Path = CACHE) -> None:
    """Portable cache: JSON for the chunks, .npy for the vectors.

    Deliberately not pickle. A pickled dataclass records the module it
    was defined in, so a cache written by `python -m app.eval.run`
    (module `__main__`) cannot be read back by a uvicorn worker, and
    the failure is an opaque AttributeError at startup rather than
    anything that names the real problem.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".vectors.npy"), np.asarray(corpus.vectors, dtype=np.float32))
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "model": corpus.model,
                "stats": corpus.stats,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "document": c.document,
                        "text": c.text,
                        "page": c.page,
                        "section": c.section,
                        "has_warning": c.has_warning,
                        "warnings": list(c.warnings),
                    }
                    for c in corpus.chunks
                ],
            }
        ),
        encoding="utf-8",
    )


def read_corpus(path: Path = CACHE) -> BuiltCorpus | None:
    import numpy as np

    meta_path = path.with_suffix(".meta.json")
    vec_path = path.with_suffix(".vectors.npy")
    if not (meta_path.exists() and vec_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vectors = np.load(vec_path)
    return BuiltCorpus(
        chunks=[IndexedChunk(**{**c, "warnings": tuple(c.get("warnings", ()))}) for c in meta["chunks"]],
        vectors=[v.tolist() for v in vectors],
        model=meta["model"],
        stats=meta["stats"],
    )


def load_or_build(rebuild: bool = False, include_catalog: bool = False) -> BuiltCorpus:
    if not rebuild:
        cached = read_corpus()
        if cached is not None:
            return cached
    corpus = build_corpus(include_catalog=include_catalog)
    save_corpus(corpus)
    return corpus


def evaluate_retrieval(
    index: InMemoryIndex, eval_set: EvalSet, embedder: LocalEmbedder, limit: int = 50
) -> RetrievalReport:
    results: list[RetrievalResult] = []
    for question in eval_set.answerable:
        vector = embedder.embed_query_sync(question.question)
        hits = index.search(question.question, vector, limit=limit)
        retrieved = [
            RetrievedChunk(
                chunk_id=h.chunk.chunk_id,
                document=h.chunk.document,
                text=h.chunk.text,
                page=h.chunk.page,
                section=h.chunk.section,
            )
            for h in hits
        ]
        rank = rank_of_hit(
            retrieved,
            question.expected_source.document,
            question.supporting_quote,
        )
        results.append(RetrievalResult(question_id=question.id, rank=rank))
    return RetrievalReport(results=results)


def evaluate_with_reranker(
    index: InMemoryIndex,
    eval_set: EvalSet,
    embedder: LocalEmbedder,
    reranker,
    retrieve_limit: int = 50,
    top_k: int = 5,
) -> tuple[RetrievalReport, RetrievalReport]:
    """Phase 2's measurement: the same questions, the same retrieved
    candidates, scored before and after reranking.

    Both reports come from one pass so the comparison cannot drift —
    reranking is applied to exactly the candidate list that the
    baseline was scored on, which is the only way the delta means
    what it claims to.
    """
    from app.query.rerank import rerank as rerank_candidates

    before: list[RetrievalResult] = []
    after: list[RetrievalResult] = []

    for question in eval_set.answerable:
        vector = embedder.embed_query_sync(question.question)
        candidates = index.search(question.question, vector, limit=retrieve_limit)

        baseline = [
            RetrievedChunk(
                chunk_id=c.chunk.chunk_id,
                document=c.chunk.document,
                text=c.chunk.text,
                page=c.chunk.page,
                section=c.chunk.section,
            )
            for c in candidates
        ]
        before.append(
            RetrievalResult(
                question_id=question.id,
                rank=rank_of_hit(
                    baseline,
                    question.expected_source.document,
                    question.supporting_quote,
                ),
            )
        )

        hits = rerank_candidates(question.question, candidates, reranker, top_k=top_k)
        reordered = [
            RetrievedChunk(
                chunk_id=h.chunk_id,
                document=h.document,
                text=h.text,
                page=h.page,
                section=h.section,
            )
            for h in hits
        ]
        after.append(
            RetrievalResult(
                question_id=question.id,
                rank=rank_of_hit(
                    reordered,
                    question.expected_source.document,
                    question.supporting_quote,
                ),
            )
        )

    return RetrievalReport(before), RetrievalReport(after)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="re-parse and re-embed")
    parser.add_argument("--include-catalog", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--rerank", action="store_true", help="phase 2: measure the delta")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="cross-encoder to measure; smaller ones trade accuracy for latency",
    )
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args()

    embedder = LocalEmbedder(cache_dir=str(ROOT / "models" / "weights"))

    print("building corpus...")
    corpus = load_or_build(rebuild=args.rebuild, include_catalog=args.include_catalog)
    for name, s in corpus.stats["documents"].items():
        print(f"  {s['chunks']:5d} chunks  {s['warned_chunks']:3d} warned  {name}")
    for name, err in corpus.stats["failed"].items():
        print(f"  FAILED {name}: {err}")
    print(f"  total {len(corpus.chunks)} chunks, embedded with {corpus.model}")

    index = InMemoryIndex()
    index.add(corpus.chunks, corpus.vectors)

    eval_set = load()
    n = len(eval_set.answerable)
    payload: dict = {"model": corpus.model, "chunks": len(corpus.chunks), "limit": args.limit}

    if not args.rerank:
        print(f"\nevaluating retrieval over {n} answerable questions")
        started = time.time()
        report = evaluate_retrieval(index, eval_set, embedder, limit=args.limit)
        elapsed = time.time() - started

        summary = report.summary()
        print(f"\n--- retrieval, top-{args.limit} ---")
        for key, value in summary.items():
            print(f"  {key:12s} {value:.3f}")
        print(f"  {'misses':12s} {len(report.misses)} / {len(report.results)}")
        print(f"  {'seconds':12s} {elapsed:.1f}")
        if report.misses:
            print(f"\n  missed: {', '.join(report.misses)}")
        payload |= {"summary": summary, "misses": report.misses}
    else:
        from app.models.reranker.local import LocalReranker

        reranker = LocalReranker(
            args.reranker_model, cache_dir=str(ROOT / "models" / "weights")
        )
        print(f"\nphase 2: retrieve top-{args.limit}, rerank to top-{args.top_k}")
        print(f"reranker: {reranker.name}")
        started = time.time()
        before, after = evaluate_with_reranker(
            index, eval_set, embedder, reranker, args.limit, args.top_k
        )
        elapsed = time.time() - started

        # Only k <= top_k is comparable: the reranked list is truncated
        # to top_k, so its recall at any deeper k is capped at recall@top_k
        # and a "drop" there would be an artefact of list length, not a
        # regression. Deeper baseline recall is reported separately as
        # headroom — the share of answers reranking could still reach.
        print(f"\n{'':12s} {'retrieval':>10s} {'reranked':>10s} {'delta':>8s}")
        for k in [k for k in (1, 3, 5, 10, 20, 50) if k <= min(args.top_k, args.limit)]:
            b, a = before.recall_at(k), after.recall_at(k)
            mark = "  <-- shown to the model" if k == args.top_k else ""
            print(f"  recall@{k:<4d} {b:10.3f} {a:10.3f} {a - b:+8.3f}{mark}")
        print(f"  {'mrr':10s} {before.mrr:10.3f} {after.mrr:10.3f} {after.mrr - before.mrr:+8.3f}")

        reachable = before.recall_at(args.limit)
        captured = after.recall_at(args.top_k)
        print(
            f"\n  headroom: retrieval put the answer in the top-{args.limit} for "
            f"{reachable:.3f}; reranking surfaced {captured:.3f} of it into the "
            f"top-{args.top_k} ({captured / reachable:.0%} of what was reachable)"
        )
        print(f"\n  {'seconds':12s} {elapsed:.1f}  ({elapsed / max(n, 1):.2f}s per query)")

        payload |= {
            "top_k": args.top_k,
            "reranker": reranker.name,
            "before": before.summary(),
            "after": after.summary(),
            "seconds_per_query": round(elapsed / max(n, 1), 3),
        }

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
