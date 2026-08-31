from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.models import RetrievedChunk

'''
chunk = RetrievedChunk(text="Docling is an open-source library for document processing.", source="sample.pdf")
embedding = [0.012, -0.045, 0.078, ...]  # 1536 numbers long
📊 QDRANT DATABASE
┌────────────────────────────────────────────────────────┐
│ ID: "a1b2-c3d4"                                        │
│ Vector: [0.012, -0.045, 0.078, ...]                    │
│ Payload: {                                             │
│   "text": "Docling is an open-source library...",     │
│   "source": "sample.pdf"                               │
│ }                                                      │
└──────────────────────
Step 2: Dense Search (search)You search for conceptual meaning: "What tools exist for reading PDFs?"You convert that query into an embedding vector query_embedding.You run search(query_embedding, top_k=1).Qdrant compares the angle between vectors (Cosine Distance).Even though the query didn't mention "Docling", the vectors are close mathematically!Return Output:Python[
    RetrievedChunk(
        text="Docling is an open-source library for document processing.",
        source="sample.pdf",
        score=0.88  # High similarity score!
    )
]
Step 3: Sparse Search (sparse_search)You search for exact keywords: "open-source library"_build_sparse_index() downloads the text payload from Qdrant and builds a keyword index (TF-IDF/BM25).It counts matching words.It finds exact word matches for "open-source" and "library".Return Output:Python[
    RetrievedChunk(
        text="Docling is an open-source library for document processing.",
        source="sample.pdf",
        score=0.95  # Exact word match!
    )
]
Step 4: Hybrid Search (hybrid_search)
What if someone searches for 
"Docling PDF features"?Dense Search finds 
context match $\rightarrow$ 
Ranks Docling as #1Sparse Search finds 
exact keyword match for 
"Docling" $\rightarrow$ Ranks Docling as #1fuse_rrf(...) 
combines both ranks ($1 + 1$) to give it a top combined score and returns it as the definitive answer!
'''

VECTOR_SIZE = 1536


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)

def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def upsert_chunks(chunks: list[RetrievedChunk], embeddings: list[list[float]]) -> None:
    ensure_collection()
    client  = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk.text, "source": chunk.source},
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)

def search(query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
    client = get_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            text=p.payload.get("text", ""),
            source=p.payload.get("source", ""),
            score=float(p.score),
        )
        for p in results
    ]


def _build_sparse_index():
    from app.services.sparse_vector_service import SparseVectorIndex
    client = get_client()
    all_points, _next_page = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    documents = [
        {
            "text": point.payload.get("text", "") if point.payload else "",
            "source": point.payload.get("source", "") if point.payload else "",
            "id": str(point.id),
        }
        for point in all_points
    ]
    sparse_index = SparseVectorIndex()
    sparse_index.fit(documents)
    return sparse_index

def sparse_search(query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Pure sparse search using TF-IDF (no dense embeddings, no fusion)."""
    sparse_index = _build_sparse_index()
    return sparse_index.search(query_text, top_k=top_k)


def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 5,
    rrf_k: int = 60,
    sparse_top_k: int = 20,
) -> list[RetrievedChunk]:

    from app.services.sparse_vector_service import fuse_rrf
    dense_results = search(query_embedding, top_k=sparse_top_k)
    sparse_index = _build_sparse_index()
    sparse_results = sparse_index.search(query_text, top_k=sparse_top_k)
    fused = fuse_rrf([dense_results, sparse_results], rrf_k=rrf_k)
    return fused[:top_k]
