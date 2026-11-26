"""Adapters for the system under test. Each returns ranked doc ids (and optionally an answer)."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Retrieval:
    ids: list[str]
    answer: str | None = None
    latency_ms: float = 0.0
    error: str | None = None


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> Retrieval: ...


def dig(obj: Any, path: str) -> Any:
    """Follow a dotted path through dicts/lists: 'data.results.0.id'. Empty path returns obj."""
    for part in filter(None, path.split(".")):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            raise KeyError(part)
    return obj


def _timed(fn: Callable[[], Retrieval]) -> Retrieval:
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as e:  # the system under test failing is a result, not a crash
        result = Retrieval(ids=[], error=f"{type(e).__name__}: {e}")
    result.latency_ms = (time.perf_counter() - start) * 1000
    return result


class HttpRetriever:
    """Calls a search/RAG HTTP endpoint.

    ``body`` is a JSON template where "{query}" and {k} are substituted, e.g.
    '{"query": "{query}", "top_k": {k}}'. ``results_path`` points at the list of hits,
    ``id_field`` at the id inside each hit (empty if hits are plain ids).
    Header values may reference environment variables: "Bearer ${API_KEY}".
    """

    def __init__(
        self,
        url: str,
        body: str = '{"query": "{query}", "k": {k}}',
        method: str = "POST",
        results_path: str = "results",
        id_field: str = "id",
        answer_path: str = "",
        headers: dict[str, str] | None = None,
        timeout: float = 30,
    ):
        self.url, self.body, self.method = url, body, method.upper()
        self.results_path, self.id_field, self.answer_path = results_path, id_field, answer_path
        self.headers = {k: os.path.expandvars(v) for k, v in (headers or {}).items()}
        self.timeout = timeout
        self.name = f"http:{url}"

    def _payload(self, query: str, k: int) -> bytes:
        quoted = json.dumps(query)[1:-1]  # escape quotes/newlines for the JSON template
        return self.body.replace("{query}", quoted).replace("{k}", str(k)).encode()

    def retrieve(self, query: str, k: int) -> Retrieval:
        def call() -> Retrieval:
            data = self._payload(query, k) if self.method != "GET" else None
            req = urllib.request.Request(self.url, data=data, method=self.method)
            req.add_header("Content-Type", "application/json")
            for key, value in self.headers.items():
                req.add_header(key, value)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    doc = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return Retrieval(ids=[], error=f"HTTP {e.code}")
            hits = dig(doc, self.results_path)
            ids = [str(dig(h, self.id_field) if self.id_field else h) for h in hits][:k]
            answer = str(dig(doc, self.answer_path)) if self.answer_path else None
            return Retrieval(ids=ids, answer=answer)

        return _timed(call)


class PythonRetriever:
    """Calls ``module:function(query, k)``. It may return a list of ids, or {"ids": [...], "answer": "..."}."""

    def __init__(self, target: str):
        module_name, _, func_name = target.partition(":")
        if not func_name:
            raise ValueError(f"expected 'module:function', got {target!r}")
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        self.fn = getattr(importlib.import_module(module_name), func_name)
        self.name = f"python:{target}"

    def retrieve(self, query: str, k: int) -> Retrieval:
        def call() -> Retrieval:
            out = self.fn(query, k)
            if isinstance(out, dict):
                return Retrieval(ids=[str(i) for i in out.get("ids", [])][:k], answer=out.get("answer"))
            return Retrieval(ids=[str(i) for i in out][:k])

        return _timed(call)


_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or the this to what when "
    "where which who why will with you your".split()
)


def _stem(word: str) -> str:
    """Just enough stemming for a fair baseline: plans→plan, costs→cost (not glass→glas)."""
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


class BM25Retriever:
    """Keyword baseline over a folder of .md/.txt files. Doc id = path relative to the folder, no extension."""

    def __init__(self, corpus_dir: str | Path, k1: float = 1.5, b: float = 0.75):
        root = Path(corpus_dir)
        files = sorted(p for p in root.rglob("*") if p.suffix in {".md", ".txt"} and p.is_file())
        if not files:
            raise ValueError(f"no .md/.txt files in {root}")
        self.ids = [str(p.relative_to(root).with_suffix("")) for p in files]
        self.docs = [Counter(tokenize(p.read_text(encoding="utf-8"))) for p in files]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b
        self.name = f"bm25:{root}"

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avgdl))
            out.append(s)
        return out

    def retrieve(self, query: str, k: int) -> Retrieval:
        def call() -> Retrieval:
            ranked = sorted(zip(self.scores(query), self.ids, strict=True), key=lambda x: -x[0])
            return Retrieval(ids=[doc_id for score, doc_id in ranked if score > 0][:k])

        return _timed(call)


def from_config(cfg: dict, base_dir: Path = Path(".")) -> Retriever:
    kind = cfg.get("type")
    if kind == "http":
        opts = {k: v for k, v in cfg.items() if k != "type"}
        return HttpRetriever(**opts)
    if kind == "python":
        return PythonRetriever(cfg["target"])
    if kind == "bm25":
        return BM25Retriever(base_dir / cfg["corpus"])
    raise ValueError(f"unknown retriever type {kind!r} (use http, python or bm25)")
