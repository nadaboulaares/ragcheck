"""ragcheck — measure any RAG or AI search system, and prove a change made it better.

    from ragcheck import load_cases, run, BM25Retriever
    result = run(load_cases("questions.jsonl"), BM25Retriever("docs/"), k=5)
    print(result.summary)
"""

from .compare import compare
from .dataset import Case, load_cases
from .retrievers import BM25Retriever, HttpRetriever, PythonRetriever, Retrieval
from .runner import Run, run

__all__ = [
    "BM25Retriever", "Case", "HttpRetriever", "PythonRetriever", "Retrieval", "Run",
    "compare", "load_cases", "run",
]
__version__ = "0.1.0"
