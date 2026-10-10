"""Loads the evaluation question set.

Two kinds of entry, both required for the set to measure anything useful:

  Answerable   question + expected answer + the document/section/page that
               supports it + a verbatim supporting quote. The quote is the
               retrieval target, and it is what lets phase 1 be measured
               before any generation exists.

  Abstention   question the system must refuse, the reason it is out of
               scope, and the nearest relevant section it should offer
               instead. A set containing only answerable questions measures
               coverage and cannot measure the property this system is
               built around.

Entries live in tests/eval_set/*.yaml, one file per drafting batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

EVAL_SET_DIR = Path(__file__).resolve().parents[2] / "tests" / "eval_set"


@dataclass(frozen=True)
class SourceRef:
    document: str
    section: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class Answerable:
    id: str
    question: str
    expected_answer: str
    expected_source: SourceRef
    supporting_quote: str
    category: str
    has_warning: bool = False
    in_table: bool = False
    in_scope: bool = True


@dataclass(frozen=True)
class Abstention:
    id: str
    question: str
    reason_out_of_scope: str
    refusal_type: str
    category: str
    nearest_relevant_section: SourceRef | None = None
    refusal_should_mention: str = ""
    in_scope: bool = False


@dataclass(frozen=True)
class EvalSet:
    answerable: list[Answerable] = field(default_factory=list)
    abstention: list[Abstention] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.answerable) + len(self.abstention)

    @property
    def ids(self) -> list[str]:
        return [q.id for q in self.answerable] + [q.id for q in self.abstention]


def _source_ref(raw: dict | None) -> SourceRef | None:
    if not raw:
        return None
    return SourceRef(
        document=raw["document"],
        section=raw.get("section"),
        page=raw.get("page"),
    )


def _entry(raw: dict) -> Answerable | Abstention:
    if raw.get("in_scope", True):
        return Answerable(
            id=raw["id"],
            question=raw["question"],
            expected_answer=raw["expected_answer"],
            expected_source=_source_ref(raw["expected_source"]),
            supporting_quote=raw["supporting_quote"],
            category=raw["category"],
            has_warning=raw.get("has_warning", False),
            in_table=raw.get("in_table", False),
        )
    return Abstention(
        id=raw["id"],
        question=raw["question"],
        reason_out_of_scope=raw["reason_out_of_scope"],
        refusal_type=raw["refusal_type"],
        category=raw["category"],
        nearest_relevant_section=_source_ref(raw.get("nearest_relevant_section")),
        refusal_should_mention=raw.get("refusal_should_mention", ""),
    )


def load(directory: Path | None = None) -> EvalSet:
    directory = directory or EVAL_SET_DIR
    answerable: list[Answerable] = []
    abstention: list[Abstention] = []
    seen: dict[str, Path] = {}

    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        for item in raw:
            entry = _entry(item)
            if entry.id in seen:
                raise ValueError(
                    f"duplicate question id {entry.id!r} in {path.name} "
                    f"and {seen[entry.id].name}"
                )
            seen[entry.id] = path
            if isinstance(entry, Answerable):
                answerable.append(entry)
            else:
                abstention.append(entry)

    return EvalSet(answerable=answerable, abstention=abstention)
