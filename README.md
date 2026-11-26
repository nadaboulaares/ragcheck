# ragcheck (`pip install retrievalcheck`)

[![CI](https://github.com/nadaboulaares/ragcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/nadaboulaares/ragcheck/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/retrievalcheck)](https://pypi.org/project/retrievalcheck/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen)

**Measure any RAG or AI search system in one command, and prove that a change made it better.**

Most AI search gets tuned by vibes: someone tries a few questions, changes the chunk size, tries again. ragcheck replaces that with numbers. Give it real questions and the documents that should be found; it runs them against your system and reports:

- **Retrieval quality**: recall@k, precision@k, hit@k, MRR, nDCG@k
- **Answer accuracy**: checks whether answers contain the facts they must contain (optional)
- **Latency**: p50 / p95 / max, measured honestly (sequential by default)
- **Per-category breakdown**: tag questions ("billing", "account"…) and see *where* the system fails, worst first
- **Before/after diffs**: which questions a change fixed, and which it broke
- **CI gate**: exit code 1 when quality drops below your bar

Zero dependencies (Python standard library only). On PyPI as **`retrievalcheck`**; the command is `ragcheck`. Works with **any** system: an HTTP endpoint, a Python function, or the built-in keyword baseline.

## Try it in 10 seconds

```bash
pip install retrievalcheck
git clone https://github.com/nadaboulaares/ragcheck && cd ragcheck
ragcheck run --config examples/acme-docs/ragcheck.toml --out runs/before.json
cd examples/acme-docs
ragcheck run --cases questions.jsonl --python better_search:search -k 3 --out ../../runs/after.json
cd ../..
ragcheck compare runs/before.json runs/after.json
```

The demo is a fictional note-taking app's help center. Plain keyword search misses questions phrased the way users actually talk ("I want to stop paying" vs. a doc titled *Cancelling your subscription*); the "improved" retriever expands queries, and `compare` shows exactly what that bought.

## Your own system

**1. Write questions** (`questions.jsonl`, one per line):

```json
{"id": "refund", "query": "Can I get my money back?", "relevant": ["billing/refunds"], "answer_contains": ["30 days"], "tags": ["billing"]}
```

`relevant` holds the ids your system returns for the right documents. `answer_contains` is optional and only checked when the system returns an answer. `tags` is optional; with it, every report adds a *By category* table:

```
| Category | Questions | Recall@3 | Hit@3 | MRR   |
|----------|----------:|---------:|------:|------:|
| features |         6 |    83.3% | 83.3% | 58.3% |   ← look here first
| billing  |         6 |    83.3% | 83.3% | 83.3% |
| account  |         3 |   100.0% | 100.0%| 100.0%|
```

**2. Point ragcheck at it.**

HTTP (anything that takes a query and returns hits):

```bash
ragcheck run --cases questions.jsonl --http http://localhost:8000/search
```

or with a config file for custom request bodies, response shapes and auth headers:

```toml
cases = "questions.jsonl"
k = 5

[retriever]
type = "http"
url = "https://api.example.com/search"
body = '{"query": "{query}", "top_k": {k}}'
results_path = "data.hits"     # dotted path to the list of hits
id_field = "document.id"       # id inside each hit ('' if hits are plain ids)
answer_path = "answer"         # optional: generated answer to check
headers = { Authorization = "Bearer ${SEARCH_API_KEY}" }   # env vars expanded

[fail_under]
recall = 0.8
latency_p95_ms = 1500          # latency thresholds are maximums
```

Python (call your retriever directly):

```python
# my_search.py
def search(query: str, k: int):
    return [hit.id for hit in my_index.query(query, top_k=k)]
    # or: return {"ids": [...], "answer": "..."}
```

```bash
ragcheck run --cases questions.jsonl --python my_search:search
```

**3. Save, change, compare.**

```bash
ragcheck run --config ragcheck.toml --out runs/before.json --html report.html
# ... change chunking / embeddings / reranker ...
ragcheck run --config ragcheck.toml --out runs/after.json
ragcheck compare runs/before.json runs/after.json --fail-on-regression
```

## In CI

```yaml
- run: pip install retrievalcheck
- run: ragcheck run --config ragcheck.toml --fail-under recall=0.85 --fail-under mrr=0.7
```

## As a library

```python
from ragcheck import load_cases, run, HttpRetriever

result = run(load_cases("questions.jsonl"), HttpRetriever("http://localhost:8000/search"), k=5)
print(result.summary)            # {'recall@5': 0.87, 'mrr': 0.74, 'latency_p95_ms': 412.0, ...}
for case in result.cases:
    if case.missed:
        print(case.query, "missed", case.missed)
```

## Metrics, briefly

| Metric | Meaning |
|---|---|
| recall@k | Share of the relevant documents that appear in the top k |
| precision@k | Share of the top k that is relevant |
| hit@k | Share of questions with at least one relevant document in the top k |
| MRR | Average of 1 / rank of the first relevant document; rewards putting the right doc first |
| nDCG@k | Rank-aware quality, 1.0 = all relevant docs at the very top |

## Need help with the numbers?

I offer a fixed-price **RAG Audit**: I build your evaluation set, find why retrieval misses, and fix it with before/after numbers. Contact: nada.boulaares@gmail.com

## License

MIT
