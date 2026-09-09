import re
import threading
from rank_bm25 import BM25Okapi
from app.models import RetrievedChunk


def technical_tokenizer(text: str) -> list[str]:
    # Splits text while preserving camelCase, snake_case, dashes, dots, and colons
    tokens = re.findall(r'[a-zA-Z0-9_\-\.\:]+', text.lower())
    return [t for t in tokens if len(t) > 1]


class SparseVectorIndex:
    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    def fit(self, documents: list[dict]) -> None:
        with self._lock:
            self.documents = documents
            if not documents:
                self.bm25 = None
                return
            corpus = [technical_tokenizer(doc.get("text", "")) for doc in documents]
            self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        with self._lock:
            if not self.bm25 or not self.documents:
                return []

            tokenized_query = technical_tokenizer(query)
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = scores.argsort()[::-1][:top_k]

            results: list[RetrievedChunk] = []
            for idx in top_indices:
                score = float(scores[idx])
                if score <= 0:
                    continue
                doc = self.documents[idx]
                results.append(
                    RetrievedChunk(
                        text=doc.get("text", ""),
                        source=doc.get("source", ""),
                        score=score,
                    )
                )
            return results


def fuse_rrf(
    result_lists: list[list[RetrievedChunk]],
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            key = chunk.text
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            if key not in meta:
                meta[key] = {"text": chunk.text, "source": chunk.source}

    return [
        RetrievedChunk(text=text, source=meta[text]["source"], score=score)
        for text, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]