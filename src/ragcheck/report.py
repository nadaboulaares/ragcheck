"""Human-readable reports: Markdown (for PRs/issues) and a standalone HTML page (for clients)."""

from __future__ import annotations

import html

from .compare import Comparison
from .runner import Run

_LABELS = {
    "cases": "Questions",
    "errors": "Errors",
    "mrr": "MRR",
    "answer_accuracy": "Answer accuracy",
    "latency_p50_ms": "Latency p50 (ms)",
    "latency_p95_ms": "Latency p95 (ms)",
    "latency_max_ms": "Latency max (ms)",
}


_NAMES = {"recall": "Recall", "precision": "Precision", "hit": "Hit", "ndcg": "nDCG"}


def _label(key: str) -> str:
    if key in _LABELS:
        return _LABELS[key]
    name, _, k = key.partition("@")
    return f"{_NAMES.get(name, name)}@{k}" if k else key


def _fmt(key: str, value) -> str:
    if value is None:
        return "—"
    if key in ("cases", "errors"):
        return str(value)
    if "latency" in key:
        return f"{value:,.2f}" if value < 10 else f"{value:,.0f}"
    return f"{value:.1%}"


def markdown(run: Run, worst: int = 10) -> str:
    s = run.summary
    lines = [
        f"# ragcheck — {run.name}",
        "",
        f"System: `{run.retriever}` · k = {run.k} · {run.created_at}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *[f"| {_label(key)} | {_fmt(key, v)} |" for key, v in s.items()],
    ]
    failing = sorted((c for c in run.cases if c.recall < 1 or c.error), key=lambda c: (c.recall, c.rr))
    if failing:
        lines += ["", f"## Questions that missed documents ({len(failing)} of {len(run.cases)})", "",
                  "| Question | Missed | Got instead |", "|---|---|---|"]
        for c in failing[:worst]:
            got = c.error or ", ".join(f"`{d}`" for d in c.retrieved[:3]) or "nothing"
            missed = ", ".join(f"`{d}`" for d in c.missed)
            lines.append(f"| {c.query.replace('|', '/')} | {missed} | {got} |")
    else:
        lines += ["", "Every question retrieved all of its relevant documents."]
    return "\n".join(lines) + "\n"


def comparison_markdown(cmp: Comparison) -> str:
    lines = [
        f"# ragcheck — {cmp.before.name} → {cmp.after.name}",
        "",
        "| Metric | Before | After | Change |",
        "|---|---:|---:|---:|",
    ]
    for key, (b, a, delta) in cmp.deltas.items():
        if delta is None:
            change = "—"
        elif "latency" in key:
            change = f"{delta:+,.2f} ms" if abs(delta) < 10 else f"{delta:+,.0f} ms"
        elif key in ("cases", "errors"):
            change = f"{delta:+d}"
        else:
            change = f"{delta * 100:+.1f} pts"
        lines.append(f"| {_label(key)} | {_fmt(key, b)} | {_fmt(key, a)} | {change} |")
    for title, items in (("Fixed", cmp.fixed), ("Regressed", cmp.regressed)):
        if items:
            lines += ["", f"## {title} ({len(items)})", ""]
            lines += [f"- {q} (recall {rb:.0%} → {ra:.0%})" for q, rb, ra in items]
    return "\n".join(lines) + "\n"


_CSS = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;max-width:960px;margin:40px auto;padding:0 20px;color:#1f2328}
h1{font-size:24px;margin-bottom:4px}.meta{color:#656d76;margin-bottom:24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:28px}
.card{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px}.card b{display:block;font-size:22px}
.card span{color:#656d76;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid #d0d7de;padding:8px;text-align:left;vertical-align:top}th{background:#f6f8fa}
code{background:#f6f8fa;padding:1px 4px;border-radius:4px;font-size:12.5px}
.bad{color:#cf222e}footer{margin-top:32px;color:#656d76;font-size:13px}
"""


def html_report(run: Run, worst: int = 25) -> str:
    s, e = run.summary, html.escape
    cards = "".join(f"<div class='card'><b>{_fmt(k, v)}</b><span>{_label(k)}</span></div>" for k, v in s.items())
    failing = sorted((c for c in run.cases if c.recall < 1 or c.error), key=lambda c: (c.recall, c.rr))
    rows = "".join(
        f"<tr><td>{e(c.query)}</td><td class='bad'>{', '.join(f'<code>{e(d)}</code>' for d in c.missed)}</td>"
        f"<td>{e(c.error) if c.error else ', '.join(f'<code>{e(d)}</code>' for d in c.retrieved[:3]) or 'nothing'}</td>"
        f"<td>{c.recall:.0%}</td></tr>"
        for c in failing[:worst]
    )
    table = (
        f"<h2>Questions that missed documents ({len(failing)} of {len(run.cases)})</h2>"
        f"<table><tr><th>Question</th><th>Missed</th><th>Got instead</th><th>Recall</th></tr>{rows}</table>"
        if failing else "<p>Every question retrieved all of its relevant documents.</p>"
    )
    return (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>ragcheck — {e(run.name)}</title>"
        f"<style>{_CSS}</style></head><body><h1>ragcheck — {e(run.name)}</h1>"
        f"<div class='meta'>System: <code>{e(run.retriever)}</code> · k = {run.k} · {e(run.created_at)}</div>"
        f"<div class='cards'>{cards}</div>{table}<footer>Generated by ragcheck</footer></body></html>\n"
    )
