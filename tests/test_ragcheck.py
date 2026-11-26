import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from ragcheck import BM25Retriever, HttpRetriever, PythonRetriever, Run, compare, load_cases, run
from ragcheck import metrics as m
from ragcheck.cli import main
from ragcheck.dataset import Case, DatasetError
from ragcheck.report import comparison_markdown, html_report, markdown
from ragcheck.retrievers import Retrieval, dig

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "acme-docs"


# --- metrics -------------------------------------------------------------------------------


def test_metrics_by_hand():
    ranked, rel = ["a", "x", "b", "y"], {"a", "b", "c"}
    assert m.recall_at_k(ranked, rel, 3) == pytest.approx(2 / 3)
    assert m.precision_at_k(ranked, rel, 3) == pytest.approx(2 / 3)
    assert m.hit_at_k(ranked, rel, 1) == 1.0 and m.hit_at_k(["x"], rel, 1) == 0.0
    assert m.reciprocal_rank(["x", "y", "b"], rel) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(["x"], rel) == 0.0
    # DCG = 1/log2(2) + 1/log2(4) ; ideal (3 relevant, k=3) = 1 + 1/log2(3) + 1/log2(4)
    expected = (1 + 0.5) / (1 + 1 / math.log2(3) + 0.5)
    assert m.ndcg_at_k(ranked, rel, 3) == pytest.approx(expected)
    assert m.ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)


def test_percentile_nearest_rank():
    assert m.percentile([], 95) == 0.0
    assert m.percentile([5.0], 50) == 5.0
    vals = list(range(1, 101))  # 1..100
    assert m.percentile(vals, 50) == 50 and m.percentile(vals, 95) == 95 and m.percentile(vals, 100) == 100


def test_answer_check():
    assert m.answer_ok("Refunds within 30 days.", ("30 days",)) is True
    assert m.answer_ok("Refunds within a month.", ("30 days",)) is False
    assert m.answer_ok(None, ("30 days",)) is None  # system gave no answer → not scored
    assert m.answer_ok("anything", ()) is None  # nothing to check


# --- dataset -------------------------------------------------------------------------------


def test_load_jsonl_and_json(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(
        "# comment\n"
        '{"query": "a?", "relevant": "doc1"}\n'
        "\n"
        '{"id": "x", "query": "b?", "relevant": ["d2", "d3"], "answer_contains": "yes"}\n'
    )
    cases = load_cases(f)
    assert cases[0] == Case("q1", "a?", ("doc1",))
    assert cases[1] == Case("x", "b?", ("d2", "d3"), ("yes",))

    j = tmp_path / "q.json"
    j.write_text(json.dumps([{"query": "c?", "relevant": ["d"]}]))
    assert load_cases(j)[0].query == "c?"


@pytest.mark.parametrize(
    "content,msg",
    [
        ('{"query": "", "relevant": ["d"]}', "no 'query'"),
        ('{"query": "a?"}', "no 'relevant'"),
        ('{"id": "1", "query": "a", "relevant": ["d"]}\n{"id": "1", "query": "b", "relevant": ["d"]}', "duplicate"),
        ("{not json", "invalid JSON"),
        ("", "no cases"),
    ],
)
def test_dataset_errors_are_clear(tmp_path, content, msg):
    f = tmp_path / "q.jsonl"
    f.write_text(content)
    with pytest.raises(DatasetError, match=msg):
        load_cases(f)


# --- retrievers ----------------------------------------------------------------------------


def test_dig():
    doc = {"data": {"hits": [{"document": {"id": 7}}]}}
    assert dig(doc, "data.hits.0.document.id") == 7
    assert dig(doc, "") is doc


def test_bm25_ranks_the_obvious_doc_first_and_handles_plurals():
    bm25 = BM25Retriever(DEMO / "corpus")
    assert bm25.retrieve("refund policy", 3).ids[0] == "billing/refunds"
    assert bm25.retrieve("How much does the Pro plan cost?", 3).ids[0] == "billing/plans"  # plan ~ plans
    assert bm25.retrieve("zzzz qqqq", 3).ids == []  # no fake matches


def test_python_retriever_list_dict_and_errors(tmp_path, monkeypatch):
    (tmp_path / "mysearch.py").write_text(
        "def plain(q, k): return ['a', 'b', 'c'][:k]\n"
        "def rich(q, k): return {'ids': ['a'], 'answer': 'hello'}\n"
        "def broken(q, k): raise RuntimeError('db down')\n"
    )
    monkeypatch.chdir(tmp_path)
    assert PythonRetriever("mysearch:plain").retrieve("q", 2).ids == ["a", "b"]
    r = PythonRetriever("mysearch:rich").retrieve("q", 5)
    assert r.ids == ["a"] and r.answer == "hello"
    bad = PythonRetriever("mysearch:broken").retrieve("q", 5)
    assert bad.ids == [] and "db down" in bad.error  # a failing system is a result, not a crash
    with pytest.raises(ValueError, match="module:function"):
        PythonRetriever("mysearch")


@pytest.fixture
def search_server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((body, self.headers.get("Authorization")))
            if body["query"] == "boom":
                self.send_response(500)
                self.end_headers()
                return
            payload = {
                "data": {"hits": [{"doc": {"id": "billing/refunds"}}, {"doc": {"id": "x"}}]},
                "answer": "Refunds are possible within 30 days.",
            }
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/search", seen
    server.shutdown()


def test_http_retriever_body_template_paths_auth_and_errors(search_server, monkeypatch):
    url, seen = search_server
    monkeypatch.setenv("SEARCH_KEY", "s3cret")
    r = HttpRetriever(
        url,
        body='{"query": "{query}", "top_k": {k}}',
        results_path="data.hits",
        id_field="doc.id",
        answer_path="answer",
        headers={"Authorization": "Bearer ${SEARCH_KEY}"},
    )
    got = r.retrieve('refund "policy"\nplease', 2)
    assert got.ids == ["billing/refunds", "x"] and "30 days" in got.answer and got.error is None
    body, auth = seen[-1]
    assert body == {"query": 'refund "policy"\nplease', "top_k": 2}  # quotes/newlines safely escaped
    assert auth == "Bearer s3cret"
    assert r.retrieve("boom", 2).error == "HTTP 500"


# --- runner, reports, compare --------------------------------------------------------------


class FakeRetriever:
    name = "fake"

    def __init__(self, answers):
        self.answers = answers

    def retrieve(self, query, k):
        ids, answer = self.answers[query]
        return Retrieval(ids=ids, answer=answer, latency_ms=10.0)


CASES = [
    Case("q1", "refund?", ("billing/refunds",), ("30 days",)),
    Case("q2", "cancel?", ("billing/cancel",)),
]


def test_run_summary_and_reports():
    fake = FakeRetriever(
        {
            "refund?": (["billing/refunds", "x"], "Within 30 days."),
            "cancel?": (["account/delete", "billing/cancel"], None),
        }
    )
    result = run(CASES, fake, k=2, name="demo")
    s = result.summary
    assert s["recall@2"] == 1.0 and s["mrr"] == pytest.approx(0.75)  # (1 + 1/2) / 2
    assert s["answer_accuracy"] == 1.0  # only q1 had an answer to check
    md = markdown(result)
    assert "| Recall@2 | 100.0% |" in md and "| MRR | 75.0% |" in md
    page = html_report(result)
    assert page.startswith("<!DOCTYPE html>") and "Every question retrieved" in page


def test_run_roundtrip_and_compare(tmp_path):
    before_sys = FakeRetriever({"refund?": (["x"], None), "cancel?": (["billing/cancel"], None)})
    after_sys = FakeRetriever({"refund?": (["billing/refunds"], None), "cancel?": (["y"], None)})
    before = run(CASES, before_sys, k=3, name="before")
    after = run(CASES, after_sys, k=3, name="after")
    before.save(tmp_path / "b.json")
    loaded = Run.load(tmp_path / "b.json")
    assert loaded.summary == before.summary

    cmp = compare(loaded, after)
    assert [q for q, *_ in cmp.fixed] == ["refund?"]
    assert [q for q, *_ in cmp.regressed] == ["cancel?"]
    assert cmp.deltas["recall@3"][2] == 0.0
    md = comparison_markdown(cmp)
    assert "## Fixed (1)" in md and "## Regressed (1)" in md

    other_k = run(CASES, FakeRetriever({"refund?": (["x"], None), "cancel?": (["y"], None)}), k=5)
    with pytest.raises(ValueError, match="different k"):
        compare(before, other_k)


def test_errors_count_and_are_reported():
    class Down:
        name = "down"

        def retrieve(self, q, k):
            return Retrieval(ids=[], error="HTTP 503")

    result = run(CASES, Down(), k=3)
    assert result.summary["errors"] == 2 and result.summary["recall@3"] == 0.0
    assert "HTTP 503" in markdown(result)


# --- CLI -----------------------------------------------------------------------------------


def test_cli_demo_end_to_end(tmp_path, monkeypatch, capsys):
    before, after, page = tmp_path / "before.json", tmp_path / "after.json", tmp_path / "r.html"
    assert main(["run", "--config", str(DEMO / "ragcheck.toml"), "--out", str(before)]) == 0
    monkeypatch.chdir(DEMO)
    assert (
        main(
            [
                "run",
                "--cases",
                "questions.jsonl",
                "--python",
                "better_search:search",
                "-k",
                "3",
                "--out",
                str(after),
                "--html",
                str(page),
            ]
        )
        == 0
    )
    assert page.read_text().startswith("<!DOCTYPE html>")
    capsys.readouterr()
    assert main(["compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out
    assert "## Fixed" in out and "Regressed" not in out


def test_cli_fail_under_gate(capsys):
    # config sets recall=0.6 (passes); CLI raises the bar → exit 1
    assert main(["run", "--config", str(DEMO / "ragcheck.toml"), "--fail-under", "recall=0.99"]) == 1
    assert "FAIL: recall@3" in capsys.readouterr().err
    assert main(["run", "--config", str(DEMO / "ragcheck.toml"), "--fail-under", "latency_p95_ms=60000"]) == 0
    assert main(["run", "--config", str(DEMO / "ragcheck.toml"), "--fail-under", "nonsense=1"]) == 1


def test_cli_usage_errors(tmp_path, capsys):
    assert main(["run", "--cases", str(DEMO / "questions.jsonl")]) == 2  # no system chosen
    assert "choose a system" in capsys.readouterr().err
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"query": "x"}')
    assert main(["run", "--cases", str(bad), "--bm25", str(DEMO / "corpus")]) == 2


# --- per-category breakdown --------------------------------------------------------------


def test_tags_parsed_deduped_and_optional(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(
        '{"query": "a", "relevant": ["d"], "tags": "billing"}\n'
        '{"query": "b", "relevant": ["d"], "tags": ["x", "billing", "x"]}\n'
        '{"query": "c", "relevant": ["d"]}\n'
    )
    assert [c.tags for c in load_cases(f)] == [("billing",), ("billing", "x"), ()]


def test_by_tag_summary_reports_and_compare():
    cases = [
        Case("q1", "refund?", ("billing/refunds",), tags=("billing",)),
        Case("q2", "cancel?", ("billing/cancel",), tags=("billing",)),
        Case("q3", "2fa?", ("account/2fa",), tags=("account",)),
    ]
    before = run(
        cases,
        FakeRetriever(
            {
                "refund?": (["billing/refunds"], None),
                "cancel?": (["x"], None),
                "2fa?": (["account/2fa"], None),
            }
        ),
        k=3,
        name="before",
    )
    tags = before.by_tag()
    assert tags["billing"]["cases"] == 2 and tags["billing"]["recall@3"] == 0.5
    assert tags["account"]["recall@3"] == 1.0

    md = markdown(before)
    # worst category first: that's where the audit should look
    assert md.index("| billing | 2 | 50.0%") < md.index("| account | 1 | 100.0%")
    assert "<h2>By category</h2>" in html_report(before)

    after = run(
        cases,
        FakeRetriever(
            {
                "refund?": (["billing/refunds"], None),
                "cancel?": (["billing/cancel"], None),
                "2fa?": (["account/2fa"], None),
            }
        ),
        k=3,
        name="after",
    )
    cmp = compare(before, after)
    assert cmp.tag_deltas["billing"] == (0.5, 1.0, 0.5)
    assert cmp.tag_deltas["account"] == (1.0, 1.0, 0.0)
    assert "| billing | 50.0% | 100.0% | +50.0 pts |" in comparison_markdown(cmp)


def test_reports_without_tags_have_no_category_section():
    result = run(CASES, FakeRetriever({"refund?": (["x"], None), "cancel?": (["y"], None)}), k=2)
    assert result.by_tag() == {}
    assert "By category" not in markdown(result) and "By category" not in html_report(result)


def test_runs_saved_before_tags_existed_still_load(tmp_path):
    old = {
        "name": "old",
        "retriever": "fake",
        "k": 3,
        "created_at": "2026-11-25T00:00:00+00:00",
        "summary": {},
        "cases": [
            {
                "id": "q1",
                "query": "a",
                "relevant": ["d"],
                "retrieved": ["d"],
                "recall": 1.0,
                "precision": 1 / 3,
                "hit": 1.0,
                "rr": 1.0,
                "ndcg": 1.0,
                "latency_ms": 1.0,
                "answer_ok": None,
                "error": None,
            }
        ],
    }
    f = tmp_path / "old.json"
    f.write_text(json.dumps(old))
    loaded = Run.load(f)
    assert loaded.cases[0].tags == [] and loaded.summary["recall@3"] == 1.0


def test_version_matches_pyproject():
    import tomllib

    import ragcheck

    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert ragcheck.__version__ == meta["version"]
    assert meta["name"] == "retrievalcheck"  # PyPI name: ragcheck is taken, ragmeter ~ rag-meter
