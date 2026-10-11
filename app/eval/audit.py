"""Separate real errors from judge strictness in a phase 3 run.

    python -m app.eval.audit

The end-to-end judge marks an answer wrong when any figure from the
expected answer is missing. That conflates two very different things:

  INCOMPLETE  every figure the answer states is correct, but it does
              not repeat something the expected answer also mentioned.
              "How long can a DBUS run be?" answered "400 m" is marked
              wrong because the expected answer also noted 4 devices
              per host - which the question never asked about.

  CONTRADICTED the answer states a figure that conflicts with the
              expected one. Answering +40 degC where the manual says
              +55 degC is a real error, and the kind that matters.

Only the second is a wrong answer. Reporting them together understates
accuracy and, worse, buries the handful of genuine errors among dozens
of harmless omissions - which is how a real defect goes unnoticed.

The split is heuristic: it compares the figures each side states and
asks whether they can be reconciled. It is a triage that tells you
which rows to read, not a verdict.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.query.verify import _MARKER_RE, _scan, _strip_list_ordinals

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "eval_phase3.json"


def _body(text: str) -> str:
    """Answer text with citation markers removed.

    A citation carries the document number — 89500A6, 89470J4 — and
    scanning it as content makes every cited answer look like it states
    figures the expected answer never mentioned. The first run of this
    audit reported 19 contradictions, almost all of which were document
    numbers.
    """
    # Step numbers are structure, not claims: a five-step procedure
    # would otherwise look like it states the figures 1 to 5 and
    # contradicts an expected answer that never mentions them.
    return _strip_list_ordinals(_MARKER_RE.sub(" ", text or ""))


def figures(text: str) -> set[str]:
    f, _, _ = _scan(_body(text))
    return {f"{x.number}{x.unit or ''}" for x in f}


def bare_numbers(text: str) -> set[str]:
    f, _, _ = _scan(_body(text))
    return {x.number for x in f}


def classify(expected: str, produced: str) -> tuple[str, list[str]]:
    want, got = figures(expected), figures(produced)
    if not want:
        return ("no-figures", [])
    missing = sorted(want - got)
    if not missing:
        return ("ok", [])
    if not got:
        return ("no-figures-given", missing)

    # A figure the answer states, carrying a unit the expected answer
    # also uses but with a different value, is a contradiction rather
    # than an omission: both sides are talking about the same quantity
    # and disagree.
    want_bare, got_bare = bare_numbers(expected), bare_numbers(produced)
    contradicting = sorted(got_bare - want_bare)
    if contradicting:
        return ("contradicted", contradicting)
    return ("incomplete", missing)


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["in_scope"] and r["answered"]]
    wrong = [r for r in rows if r.get("values_ok") is False]

    buckets: dict[str, list[dict]] = {}
    for r in wrong:
        kind, detail = classify(r["expected"], r["answer"])
        r["_kind"], r["_detail"] = kind, detail
        buckets.setdefault(kind, []).append(r)

    total_answered = len(rows)
    counts = Counter(r["_kind"] for r in wrong)
    real = counts.get("contradicted", 0)
    harmless = len(wrong) - real

    print("=" * 62)
    print("PHASE 3 AUDIT — what the judge called wrong")
    print("=" * 62)
    print(f"  answered in scope        {total_answered}")
    print(f"  judged wrong             {len(wrong)}")
    for kind, n in counts.most_common():
        print(f"     {kind:<20} {n}")
    print()
    print(f"  CONTRADICTED (real)      {real}")
    print(f"  omission / strictness    {harmless}")
    print()
    strict = (total_answered - len(wrong)) / total_answered
    lenient = (total_answered - real) / total_answered
    print(f"  accuracy, strict judge   {strict:.3f}")
    print(f"  accuracy, contradictions only {lenient:.3f}   <- the defensible figure")

    for kind in ("contradicted", "no-figures-given", "incomplete"):
        rs = buckets.get(kind, [])
        if not rs:
            continue
        print(f"\n--- {kind.upper()} ({len(rs)}) ---")
        for r in rs[:8]:
            print(f"  {r['qid']}: {r['question'][:66]}")
            print(f"     expected: {r['expected'][:88]}")
            print(f"     got     : {(r['answer'] or '')[:88].replace(chr(10), ' | ')}")
            if kind == "contradicted":
                print(f"     conflicting figures: {r['_detail'][:6]}")
            print()


if __name__ == "__main__":
    main()
