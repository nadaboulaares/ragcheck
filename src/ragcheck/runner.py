"""Run every case against a retriever and aggregate the results."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import metrics as m
from .dataset import Case
from .retrievers import Retriever


@dataclass
class CaseResult:
    id: str
    query: str
    relevant: list[str]
    retrieved: list[str]
    recall: float
    precision: float
    hit: float
    rr: float
    ndcg: float
    latency_ms: float
    answer_ok: bool | None = None
    error: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def missed(self) -> list[str]:
        return [d for d in self.relevant if d not in self.retrieved]


@dataclass
class Run:
    name: str
    retriever: str
    k: int
    created_at: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        return summarize(self.cases, self.k)

    def by_tag(self) -> dict[str, dict]:
        """Summary per tag, so a report can say *where* the system fails ("billing: 60% recall")."""
        tags = sorted({t for c in self.cases for t in c.tags})
        return {t: summarize([c for c in self.cases if t in c.tags], self.k) for t in tags}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "retriever": self.retriever,
            "k": self.k,
            "created_at": self.created_at,
            "summary": self.summary,
            "by_tag": self.by_tag(),
            "cases": [asdict(c) for c in self.cases],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Run:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        run = cls(d["name"], d["retriever"], d["k"], d["created_at"])
        run.cases = [CaseResult(**c) for c in d["cases"]]
        return run


def summarize(cases: list[CaseResult], k: int) -> dict:
    ok = [c for c in cases if c.error is None]
    n = len(cases) or 1

    def mean(attr: str) -> float:
        return sum(getattr(c, attr) for c in cases) / n

    answers = [c.answer_ok for c in cases if c.answer_ok is not None]
    lat = [c.latency_ms for c in ok]
    return {
        "cases": len(cases),
        "errors": len(cases) - len(ok),
        f"recall@{k}": round(mean("recall"), 4),
        f"precision@{k}": round(mean("precision"), 4),
        f"hit@{k}": round(mean("hit"), 4),
        "mrr": round(mean("rr"), 4),
        f"ndcg@{k}": round(mean("ndcg"), 4),
        "answer_accuracy": round(sum(answers) / len(answers), 4) if answers else None,
        "latency_p50_ms": round(m.percentile(lat, 50), 3),
        "latency_p95_ms": round(m.percentile(lat, 95), 3),
        "latency_max_ms": round(max(lat), 3) if lat else 0.0,
    }


def evaluate_case(case: Case, retriever: Retriever, k: int) -> CaseResult:
    r = retriever.retrieve(case.query, k)
    ranked, relevant = r.ids[:k], set(case.relevant)
    return CaseResult(
        id=case.id,
        query=case.query,
        relevant=list(case.relevant),
        retrieved=ranked,
        recall=m.recall_at_k(ranked, relevant, k),
        precision=m.precision_at_k(ranked, relevant, k),
        hit=m.hit_at_k(ranked, relevant, k),
        rr=m.reciprocal_rank(ranked, relevant),
        ndcg=m.ndcg_at_k(ranked, relevant, k),
        latency_ms=round(r.latency_ms, 3),
        answer_ok=m.answer_ok(r.answer, case.answer_contains),
        error=r.error,
        tags=list(case.tags),
    )


def run(cases: list[Case], retriever: Retriever, k: int = 5, name: str = "run", concurrency: int = 1) -> Run:
    """Sequential by default so latency numbers aren't distorted by self-inflicted load."""
    result = Run(name, retriever.name, k, datetime.now(UTC).isoformat(timespec="seconds"))
    if concurrency <= 1:
        result.cases = [evaluate_case(c, retriever, k) for c in cases]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            result.cases = list(pool.map(lambda c: evaluate_case(c, retriever, k), cases))
    return result
