"""ragcheck command line.

ragcheck run --cases questions.jsonl --bm25 docs/            # keyword baseline
ragcheck run --cases questions.jsonl --http http://localhost:8000/search
ragcheck run --config ragcheck.toml --out runs/after.json --fail-under recall=0.8
ragcheck compare runs/before.json runs/after.json
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from . import report
from .compare import compare
from .dataset import DatasetError, load_cases
from .retrievers import BM25Retriever, HttpRetriever, PythonRetriever, from_config
from .runner import Run, run


def _parse_thresholds(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--fail-under expects metric=value, got {item!r}")
        out[key.strip().lower()] = float(value)
    return out


def _check_thresholds(summary: dict, thresholds: dict[str, float], k: int) -> list[str]:
    """Metric names may omit '@k' ('recall' means recall@k). Latency thresholds are maximums."""
    failures = []
    for key, limit in thresholds.items():
        name = key if key in summary else f"{key}@{k}"
        if name not in summary or summary[name] is None:
            failures.append(f"unknown or empty metric {key!r}")
        elif "latency" in name:
            if summary[name] > limit:
                failures.append(f"{name} = {summary[name]} > {limit}")
        elif summary[name] < limit:
            failures.append(f"{name} = {summary[name]} < {limit}")
    return failures


def cmd_run(args) -> int:
    cfg, base = {}, Path(".")
    if args.config:
        cfg_path = Path(args.config)
        cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        base = cfg_path.parent

    cases_path = args.cases or (base / cfg["cases"] if "cases" in cfg else None)
    if not cases_path:
        print("error: give --cases or set 'cases' in the config", file=sys.stderr)
        return 2
    try:
        cases = load_cases(cases_path)
    except (DatasetError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.http:
        retriever = HttpRetriever(args.http, results_path=args.results_path, id_field=args.id_field)
    elif args.python:
        retriever = PythonRetriever(args.python)
    elif args.bm25:
        retriever = BM25Retriever(args.bm25)
    elif "retriever" in cfg:
        retriever = from_config(cfg["retriever"], base)
    else:
        print(
            "error: choose a system: --http URL, --python module:func, --bm25 DIR, or [retriever] in the config",
            file=sys.stderr,
        )
        return 2

    k = args.k or cfg.get("k", 5)
    name = args.name or (Path(args.out).stem if args.out else "run")
    result = run(cases, retriever, k=k, name=name, concurrency=args.concurrency)

    text = report.markdown(result)
    print(text)
    if args.out:
        result.save(args.out)
        print(f"saved {args.out}", file=sys.stderr)
    if args.html:
        Path(args.html).write_text(report.html_report(result), encoding="utf-8")
        print(f"saved {args.html}", file=sys.stderr)

    # Config thresholds first; command-line --fail-under overrides them.
    thresholds = {key.lower(): float(v) for key, v in cfg.get("fail_under", {}).items()}
    thresholds |= _parse_thresholds(args.fail_under)
    failures = _check_thresholds(result.summary, thresholds, k)
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    return 1 if failures else 0


def cmd_compare(args) -> int:
    try:
        cmp = compare(Run.load(args.before), Run.load(args.after))
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(report.comparison_markdown(cmp))
    return 1 if (args.fail_on_regression and cmp.regressed) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure any RAG / AI search system.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate a system against a question set")
    r.add_argument("--config", help="TOML config (see examples/)")
    r.add_argument("--cases", help="questions file (.jsonl or .json)")
    src = r.add_mutually_exclusive_group()
    src.add_argument("--http", metavar="URL", help="POST {query, k} to this endpoint")
    src.add_argument("--python", metavar="MODULE:FUNC", help="call func(query, k)")
    src.add_argument("--bm25", metavar="DIR", help="built-in keyword baseline over .md/.txt files")
    r.add_argument("--results-path", default="results", help="(http) dotted path to the hit list")
    r.add_argument("--id-field", default="id", help="(http) id field inside each hit; '' for plain ids")
    r.add_argument("-k", type=int, help="top-k to evaluate (default 5)")
    r.add_argument("--name", help="run name shown in reports")
    r.add_argument("--out", help="save the full run as JSON (for compare)")
    r.add_argument("--html", help="also write a standalone HTML report")
    r.add_argument("--concurrency", type=int, default=1, help="parallel requests (default 1: honest latency)")
    r.add_argument(
        "--fail-under",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="exit 1 if a metric misses the bar, e.g. recall=0.8, latency_p95_ms=1500",
    )
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="before/after diff of two saved runs")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any question got worse")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
