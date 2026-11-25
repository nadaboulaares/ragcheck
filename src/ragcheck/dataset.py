"""Evaluation cases: a question plus the documents a good system must find."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    relevant: tuple[str, ...]  # doc ids that should be retrieved
    answer_contains: tuple[str, ...] = field(default=())  # optional: phrases a correct answer includes


class DatasetError(ValueError):
    pass


def load_cases(path: str | Path) -> list[Case]:
    """Load cases from JSONL (one object per line) or a JSON list.

    Each object: {"query": "...", "relevant": ["doc-a", ...], "id"?: "...", "answer_contains"?: [...]}
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise DatasetError(f"{path}: expected a JSON list of cases")
    else:
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip() and not line.lstrip().startswith("#"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetError(f"{path}:{n}: invalid JSON ({e.msg})") from e

    cases, seen = [], set()
    for n, row in enumerate(rows, 1):
        query = (row.get("query") or "").strip()
        relevant = row.get("relevant") or []
        if isinstance(relevant, str):
            relevant = [relevant]
        if not query:
            raise DatasetError(f"{path}: case {n} has no 'query'")
        if not relevant:
            raise DatasetError(f"{path}: case {n} ({query[:40]!r}) has no 'relevant' documents")
        case_id = str(row.get("id") or f"q{n}")
        if case_id in seen:
            raise DatasetError(f"{path}: duplicate case id {case_id!r}")
        seen.add(case_id)
        answer = row.get("answer_contains") or []
        if isinstance(answer, str):
            answer = [answer]
        cases.append(Case(case_id, query, tuple(map(str, relevant)), tuple(map(str, answer))))
    if not cases:
        raise DatasetError(f"{path}: no cases found")
    return cases
