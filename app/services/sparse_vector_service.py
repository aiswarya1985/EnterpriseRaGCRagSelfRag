import threading

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import RetrievedChunk
'''
self.documents we are saying list of dict
lock to prevebt deadlock when multiple threads are accessing the index
'''
class SparseVectorIndex:

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents: list[dict] = []
        self.matrix = None
        self._lock = threading.RLock()

    '''
    documents = [
        {"text": "Python search engine", "source": "doc1.txt"},
        {"text": "Hybrid sparse search", "source": "doc2.txt"}
       self.matrix will have 2 rows and 5 columns
       [python ,search, engine, hybrid, sparse]
       [
        [0.577, 0.000, 0.577, 0.408, 0.000],  # Doc 1 numbers ("Python search engine")
        [0.000, 0.630, 0.000, 0.447, 0.630]   # Doc 2 numbers ("Hybrid sparse search")
     ]
    ]''' 
    def fit(self, documents: list[dict]) -> None:
        with self._lock:
            self.documents = documents
            if not documents:
                self.matrix = None
                return

            texts = [doc.get("text", "") for doc in documents]
            try:
                self.matrix = self.vectorizer.fit_transform(texts)
            except ValueError:
                self.matrix = None
                return

            if self.matrix.shape[1] == 0:
                self.matrix = None

    '''
    * **Vocabulary mapping:** Your 2 documents generate a global 5-word vocabulary (`engine`, `hybrid`, `python`, `search`, `sparse`).
    * **Query vectorization:** `"python search"` converts to `[0, 0, 0.71, 0.71, 0]`, where `0.71` comes from $1 / \sqrt{2}$ L2 normalization.
    * **Matrix multiplication:** `cosine_similarity` computes dot products between the query vector and every matrix row; non-matching terms multiply by `0` and drop out.
    * **Score evaluation:** Doc 0 scores `0.70` (matches `"python"` and `"search"`), while Doc 1 scores `0.32` (matches `"search"` only).
    * **Ranking & output:** `argsort()[::-1]` sorts scores in descending order (`[0, 1]`), allowing `search()` to fetch and return the highest-scoring `RetrievedChunk` objects first.
    '''
    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Return top-k chunks by TF-IDF cosine similarity."""
        with self._lock:
            if self.matrix is None or len(self.documents) == 0:
                return []

            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]

            results: list[RetrievedChunk] = []
            for idx in top_indices:
                score = float(similarities[idx])
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



        '''
        result_lists = [
        [Chunk_A, Chunk_B],  # List 0: Sparse results
        [Chunk_A, Chunk_C]   # List 1: Dense results
        ]
        Document,Sparse Rank Score,Dense Rank Score,Total Fused Score
        "Doc 1 (""Python search engine"")",1/(60+0+1)=0.01639,1/(60+0+1)=0.01639,0.01639+0.01639=0.03278
        "Doc 2 (""Hybrid sparse search"")",1/(60+1+1)=0.01613,1/(60+1+1)=0.01613,0.01613+0.01613=0.03226
        '''




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