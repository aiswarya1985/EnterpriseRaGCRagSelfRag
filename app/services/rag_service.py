from __future__ import annotations

from loguru import logger

from app.config import settings
from app.models import (
    ChatResponse,
    ResponseMetadata,
    RetrievedChunk,
    RetrievedChunkPreview,
)
from app.security.spotlighting import build_spotlighted_context
from app.security.system_prompt import build_system_prompt
from app.services.embedding_service import embed_texts
from app.services.llm_service import generate
from app.services.vector_store import search, hybrid_search, sparse_search
from app.services.query_cache_service import query_cache


'''[[0.012, -0.045, ...]] we will take the inner list
that is why embeddings[0]'''
def _retrieve(question: str, top_k: int=5) -> list[RetrievedChunk]:
   embeddings=embed_texts([question])
   return search(embeddings[0],top_k=top_k)

'''{
    "text": "Based on the provided context, the answer is...",
    "tokens": 145,
    "model": "gpt-4",
    "finish_reason": "stop"
}'''
def _generate(question:str,chunks:list[RetrievedChunk])->ChatResponse:
    spotlighted=build_spotlighted_context(chunks)
    system=build_system_prompt()
    user_msg=f"{spotlighted}\n\nQuestion:{question}"
    raw=generate(system,user_msg)["text"]
    chunk_previews = [
        RetrievedChunkPreview(text=c.text, source=c.source, score=c.score) for c in chunks
    ]
    return ChatResponse(
        answer=raw,
        sources=list({c.source for c in chunks}),
        confidence=0.7,
        metadata=ResponseMetadata(
            route="rag",
            retrieved_chunks=chunk_previews            
        ),
    )

def _flag_top_k_from_flags(flags:dict | int| None)-> int:
    if flags is None:
        return 5       
    if isinstance(flags, int):
        return flags
    return int(flags.get("top_k", 5))
 
def run_rag(question:str, flags:dict | int| None=None)->ChatResponse:
    top_k=_flag_top_k_from_flags(flags)
    logger.info(f"Running RAG for question: {question} with top_k={top_k}")
    chunks=_retrieve(question,top_k)
    return _generate(question,chunks)


def run_rag_with_trace(question:str, flags:dict | int| None=None)->tuple[ChatResponse, list[RetrievedChunk]]:
    top_k=_flag_top_k_from_flags(flags)
    logger.info(f"Running RAG with trace for question: {question} with top_k={top_k}")
    chunks=_retrieve(question,top_k)
    response=_generate(question,chunks)
    return response,chunks


run_rag_with_trace_no_cache = run_rag_with_trace

