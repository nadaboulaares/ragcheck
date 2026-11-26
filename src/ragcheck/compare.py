"""Before/after comparison of two runs on the same questions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .runner import Run


@dataclass
class Comparison:
    before: Run
    after: Run
    deltas: dict  # metric -> (before, after, after - before)
    fixed: list  # (query, recall_before, recall_after) where recall went up
    regressed: list  # ... where recall went down
    tag_deltas: dict = field(default_factory=dict)  # tag -> (recall_before, recall_after, change)


def compare(before: Run, after: Run) -> Comparison:
    if before.k != after.k:
        raise ValueError(f"runs use different k ({before.k} vs {after.k}); metrics aren't comparable")
    sb, sa = before.summary, after.summary
    deltas = {}
    for key in sb:
        b, a = sb[key], sa.get(key)
        deltas[key] = (b, a, None if b is None or a is None else round(a - b, 4))

    after_by_id = {c.id: c for c in after.cases}
    fixed, regressed = [], []
    for c in before.cases:
        other = after_by_id.get(c.id)
        if other is None:
            continue
        if other.recall > c.recall:
            fixed.append((c.query, c.recall, other.recall))
        elif other.recall < c.recall:
            regressed.append((c.query, c.recall, other.recall))
    key = f"recall@{before.k}"
    tb, ta = before.by_tag(), after.by_tag()
    tag_deltas = {}
    for tag in sorted(set(tb) | set(ta)):
        b = tb.get(tag, {}).get(key)
        a = ta.get(tag, {}).get(key)
        tag_deltas[tag] = (b, a, None if b is None or a is None else round(a - b, 4))
    return Comparison(before, after, deltas, fixed, regressed, tag_deltas)
