"""An 'improved' retriever for the demo: same keyword search plus query expansion.

Real systems would add embeddings / hybrid search / reranking; this just shows how
ragcheck proves (or disproves) that a change helped.
"""
from pathlib import Path

from ragcheck import BM25Retriever

_HERE = Path(__file__).parent
_bm25 = BM25Retriever(_HERE / "corpus")

# How users talk -> how the docs talk.
SYNONYMS = {
    "money back": "refund", "stop paying": "cancel subscription", "receipt": "invoice",
    "log in": "sign-in", "lost my phone": "recovery codes two-factor", "close": "delete",
    "remove": "delete", "back up": "export", "wifi": "offline internet connection",
    "plane": "offline", "coworker": "teammate share", "upload size": "attachment",
    "by mistake": "trash restore", "get it back": "restore",
}


def search(query: str, k: int) -> list[str]:
    q = query.lower()
    expanded = query + " " + " ".join(v for phrase, v in SYNONYMS.items() if phrase in q)
    return _bm25.retrieve(expanded, k).ids
