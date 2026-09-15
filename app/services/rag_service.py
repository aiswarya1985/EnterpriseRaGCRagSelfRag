from __future__ import annotations

import asyncio

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
from app.services.crag import crag_pipeline
from app.services.crag import crag_pipeline
from app.services.embedding_service import embed_texts
from app.services.hyde import HyDERetriever
from app.services.llm_service import generate
from app.services.self_reflective import reflect_on_answer, should_regenerate
from app.services.vector_store import search, hybrid_search, sparse_search
from app.services.query_cache_service import query_cache
from app.services.reranking import Reranker

#region flags:
def _flag(flags: dict | None, key: str, default):
    if not isinstance(flags, dict):
        return default
    return flags.get(key, default)
#endregion

async def _retrieve(question: str, flags: dict | None = None) -> list[RetrievedChunk]:
    logger.info(f"flags: {flags}")
    final_top_k = int(_flag(flags, "top_k", 5))
    mode = _flag(flags, "search_mode", "dense")
    rerank = bool(_flag(flags, "rerank", False))
    hyde = bool(_flag(flags, "hyde", False))
    enable_crag = bool(_flag(flags, "crag", settings.crag_enabled_by_default))
    
    retrieve_k = settings.reranker_initial_top_k if rerank else final_top_k
    logger.info(f"final flags: top_k={final_top_k}, search_mode={mode}, enable_rerank={rerank}, enable_hyde={hyde}, enable_crag={enable_crag}")
    
    if mode == "sparse":
        chunks = sparse_search(question, top_k=retrieve_k)
    elif mode == "hybrid" and not hyde:
        query_embedding = embed_texts([question])[0]
        chunks = hybrid_search(query_embedding, question, top_k=retrieve_k)
    elif hyde and mode == "hybrid":
        chunks = await HyDERetriever().retrieve(question, top_k=retrieve_k)
    else:
        query_embedding = embed_texts([question])[0]
        logger.info(f"Query embedding sample: {len(query_embedding)}")
        chunks = search(query_embedding, top_k=retrieve_k)
        logger.info(f"Retrieved chunks: {chunks[0] if chunks else 'No chunks retrieved'}")

    if rerank and chunks:
        logger.info("reranking true")
        reranker = Reranker()
        chunks = reranker.rerank(question, chunks, top_k=final_top_k)
    else:
        chunks = chunks[:final_top_k]

    if enable_crag:
     # CRAG: grade chunks + fall back to web search if irrelevant
        chunks, evaluation, used_web = crag_pipeline(
            question=question,
            chunks=chunks,
            enable_crag=enable_crag,
    )
        logger.info(
            "CRAG | enabled={} score={} label={} used_web={}",
            enable_crag,
            evaluation.relevance_score,
            evaluation.relevance_label,
            used_web,
    )    

    return chunks

def _generate(
    question: str,
    chunks: list[RetrievedChunk],
    flags: dict | None = None,
) -> ChatResponse:
    enable_self_reflective = bool(_flag(flags, "srag", False))

    spotlighted = build_spotlighted_context(chunks)
    system = build_system_prompt()

    def _raw(q: str) -> str:
        return generate(system, f"{spotlighted}\n\nQuestion: {q}")["text"]

    working_q = question
    raw = _raw(working_q)

    # Self-RAG: reflect on the answer; refine the question and retry if weak.
    iterations = 0
    last_score: float | None = None
    final_refined: str | None = None
    if enable_self_reflective:
        while True:
            reflection = reflect_on_answer(
                question=working_q,
                answer=raw,
                context=spotlighted,
            )
            last_score = float(reflection.reflection_score)
            logger.info(
                "Self-RAG | iteration={} score={} needs_regeneration={}",
                iterations,
                last_score,
                reflection.needs_regeneration,
            )
            final_refined = reflection.refined_question or working_q
            if not should_regenerate(reflection, iterations):
                break           
           
            working_q = final_refined
            raw = _raw(working_q)
            iterations += 1

    chunk_previews = [
        RetrievedChunkPreview(text=c.text, source=c.source, score=c.score) for c in chunks
    ]
    return ChatResponse(
        answer=raw,
        sources=list({c.source for c in chunks}),
        confidence=0.7,
        metadata=ResponseMetadata(
            route="rag",
            retrieved_chunks=chunk_previews,
            reflection_iterations=iterations,
            reflection_score=last_score,
            refined_question=final_refined,
        ),
    )

async def run_rag_async(question: str, flags: dict | int | None = None) -> ChatResponse:
    logger.info(f"Running RAG with question: {question}, flags: {flags}")
    chunks = await _retrieve(question, flags=flags if isinstance(flags, dict) else None)
    response = _generate(question, chunks, flags=flags if isinstance(flags, dict) else None)
    return response

def run_rag(question: str, flags: dict | int | None = None) -> ChatResponse:
    return asyncio.run(run_rag_async(question, flags))

def run_rag_with_trace(
    question: str, flags: dict | int | None = None
) -> tuple[ChatResponse, list[RetrievedChunk]]:
    
    chunks = asyncio.run(_retrieve(question, flags=flags if isinstance(flags, dict) else None))
    response = _generate(question, chunks, flags=flags if isinstance(flags, dict) else None)
    return response, chunks


run_rag_with_trace_no_cache = run_rag_with_trace
