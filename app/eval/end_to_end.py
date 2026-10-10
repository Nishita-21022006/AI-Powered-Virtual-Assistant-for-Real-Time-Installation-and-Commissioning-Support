"""Phase 3 evaluation: the whole pipeline, all 104 questions.

Produces the four numbers the project exists to justify:

  answer rate            of the in-scope questions, how many it answers
  value accuracy         of those, how many state the expected figures
  abstention precision   of the out-of-scope questions, how many it refuses
  false answer rate      answers given that were wrong, or given at all
                         on an out-of-scope question

The last one is the number that matters. The target shape is ~85%
answered at ~99% accuracy, not maximum coverage — an assistant that
answers everything at 90% is worse than one that answers 85% at 99%,
because a confidently wrong wiring answer ends the project.

HOW CORRECTNESS IS JUDGED, and the limits of it
-----------------------------------------------
Deterministically, by reusing the verifier's figure and identifier
extraction: every number, measurement and part code in the expected
answer must appear in the generated one. No model judges another model
here, so the result is reproducible and cannot drift.

The cost is that it is strict in one direction and blind in another. It
cannot tell whether prose is well written, and it cannot catch an
answer that contains all the right numbers arranged into a wrong claim.
For an expected answer carrying no figures at all it falls back to
keyword overlap, which is weaker. Treat value accuracy as a floor on
quality rather than a measure of it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.eval.dataset import Abstention, Answerable, load
from app.eval.metrics import AnswerOutcome, AnswerReport
from app.query.pipeline import Outcome, QueryPipeline
from app.query.verify import _scan

ROOT = Path(__file__).resolve().parents[2]

# Groq free tier is ~30 requests/minute. One call per question plus
# headroom keeps us under it without a retry loop.
THROTTLE_SECONDS = 2.2


@dataclass
class Row:
    qid: str
    question: str
    in_scope: bool
    outcome: str
    answered: bool
    answer: str = ""
    expected: str = ""
    values_ok: bool | None = None
    cited_expected_doc: bool | None = None
    missing_values: list[str] = field(default_factory=list)
    reason: str = ""


def _values(text: str) -> set[str]:
    figures, identifiers, _ = _scan(text)
    return {f"{f.number}{f.unit or ''}" for f in figures} | {
        i.canonical for i in identifiers
    }


def _keywords(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for",
        "with", "be", "must", "can", "it", "this", "that", "no", "not", "from",
    }
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if len(w) > 3 and w not in stop}


def judge(expected: str, produced: str) -> tuple[bool, list[str]]:
    """Every figure and identifier in the expected answer must appear."""
    want = _values(expected)
    if want:
        got = _values(produced)
        missing = sorted(want - got)
        return not missing, missing
    # No figures to check — fall back to keyword overlap, which is a much
    # weaker signal and is flagged as such in the report.
    want_words = _keywords(expected)
    if not want_words:
        return True, []
    overlap = len(want_words & _keywords(produced)) / len(want_words)
    return overlap >= 0.4, [] if overlap >= 0.4 else ["low keyword overlap"]


def run(pipeline: QueryPipeline, limit: int | None = None) -> tuple[list[Row], AnswerReport]:
    eval_set = load()
    questions: list[Answerable | Abstention] = [
        *eval_set.answerable,
        *eval_set.abstention,
    ]
    if limit:
        questions = questions[:limit]

    rows: list[Row] = []
    outcomes: list[AnswerOutcome] = []

    for n, q in enumerate(questions, start=1):
        in_scope = isinstance(q, Answerable)
        result = pipeline.run(q.question)
        answered = result.outcome is Outcome.ANSWERED

        row = Row(
            qid=q.id,
            question=q.question,
            in_scope=in_scope,
            outcome=str(result.outcome),
            answered=answered,
            answer=(result.answer or "")[:600],
            reason=result.reason[:160],
        )

        correct: bool | None = None
        if in_scope:
            row.expected = q.expected_answer
            if answered:
                ok, missing = judge(q.expected_answer, result.answer or "")
                row.values_ok, row.missing_values, correct = ok, missing, ok
                row.cited_expected_doc = any(
                    p.document == q.expected_source.document for p in result.passages
                )
        rows.append(row)
        outcomes.append(
            AnswerOutcome(
                question_id=q.id,
                answered=answered,
                correct=correct,
                should_have_answered=in_scope,
            )
        )
        print(
            f"  [{n:3d}/{len(questions)}] {q.id:5s} "
            f"{'in ' if in_scope else 'out'} {str(result.outcome):26s} "
            f"{'' if correct is None else ('OK' if correct else 'WRONG')}",
            flush=True,
        )
        time.sleep(THROTTLE_SECONDS)

    return rows, AnswerReport(outcomes)


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", type=Path, default=Path("eval_phase3.json"))
    args = parser.parse_args()

    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    from app.eval.run import read_corpus
    from app.models.embedder.local import LocalEmbedder
    from app.models.generator.api import ApiGenerator
    from app.models.reranker.local import LocalReranker
    from app.query.retrieve import InMemoryIndex

    weights = str(ROOT / "models" / "weights")
    corpus = read_corpus()
    index = InMemoryIndex()
    index.add(corpus.chunks, corpus.vectors)

    pipeline = QueryPipeline(
        index=index,
        embedder=LocalEmbedder(cache_dir=weights),
        reranker=LocalReranker(os.environ["RERANKER_MODEL"], cache_dir=weights),
        generator=ApiGenerator(
            os.environ["GENERATOR_MODEL"],
            os.environ["GROQ_API_KEY"],
            os.environ["GROQ_BASE_URL"],
        ),
        score_floor=float(os.environ.get("RETRIEVAL_SCORE_FLOOR", "-5.0")),
        show_sources_floor=float(os.environ.get("SHOW_SOURCES_FLOOR", "-8.0")),
    )

    started = time.time()
    rows, report = run(pipeline, limit=args.limit)
    elapsed = time.time() - started

    summary = report.summary()
    in_scope = [r for r in rows if r.in_scope]
    answered = [r for r in in_scope if r.answered]
    wrong = [r for r in answered if r.values_ok is False]
    leaked = [r for r in rows if not r.in_scope and r.answered]
    cited = [r for r in answered if r.cited_expected_doc]

    print("\n" + "=" * 62)
    print("PHASE 3 — END TO END")
    print("=" * 62)
    print(f"  answer rate            {summary['answer_rate']:.3f}"
          f"   ({len(answered)}/{len(in_scope)} in-scope answered)")
    print(f"  value accuracy         {summary['accuracy']:.3f}"
          f"   ({len(answered) - len(wrong)}/{len(answered)} of answers)")
    print(f"  abstention precision   {summary['abstention_precision']:.3f}"
          f"   ({len(rows) - len(in_scope) - len(leaked)}/{len(rows) - len(in_scope)} refused)")
    print(f"  FALSE ANSWER RATE      {summary['false_answer_rate']:.3f}"
          f"   ({len(wrong) + len(leaked)}/{len(rows)}) <-- the number that matters")
    print(f"\n  cited the expected doc {len(cited)}/{len(answered)}")
    print(f"  elapsed                {elapsed / 60:.1f} min")

    if wrong:
        print(f"\n  WRONG VALUES ({len(wrong)}):")
        for r in wrong[:10]:
            print(f"    {r.qid}: missing {r.missing_values[:4]}")
    if leaked:
        print(f"\n  ANSWERED WHEN IT SHOULD HAVE REFUSED ({len(leaked)}):")
        for r in leaked[:10]:
            print(f"    {r.qid}: {r.question[:70]}")

    args.json.write_text(
        json.dumps(
            {
                "summary": summary,
                "counts": {
                    "in_scope": len(in_scope),
                    "answered": len(answered),
                    "wrong_values": len(wrong),
                    "leaked": len(leaked),
                    "cited_expected_doc": len(cited),
                },
                "rows": [r.__dict__ for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  written to {args.json}")


if __name__ == "__main__":
    main()
