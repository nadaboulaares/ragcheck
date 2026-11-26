# Changelog

## 0.2.0 — first PyPI release as `ragmeter`

- Published on PyPI as **`ragmeter`** (`ragcheck` was taken). Commands `ragmeter` and `ragcheck` both work;
  the import name stays `ragcheck`.

- **Per-category breakdown.** Questions can carry `"tags": [...]`; every report adds a *By category* table
  (worst first), and `compare` shows recall before/after per category. Runs saved without tags still load.
- Lint (ruff) and formatting enforced; CI on Python 3.11–3.13 with an end-to-end demo smoke test.

## 0.1.0 — 2026-11-25

- First version: recall@k, precision@k, hit@k, MRR, nDCG@k, answer checks, latency p50/p95/max.
- Systems: HTTP endpoints (body templates, dotted response paths, env-expanded auth headers),
  Python functions, built-in BM25 baseline.
- Markdown, HTML and JSON reports; `compare` with fixed/regressed questions; `--fail-under` CI gate.
