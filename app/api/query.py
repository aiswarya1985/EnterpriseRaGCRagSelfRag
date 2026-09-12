from fastapi import APIRouter, Depends
from app.middleware.auth import User, get_current_user
from app.models import ChatResponse, QueryRequest
from app.services.rag_service import run_rag_async
from loguru import logger

router=APIRouter(tags=["query"])

@router.post("/query", response_model=ChatResponse)
async def query(
    body: QueryRequest, 
    user: User = Depends(get_current_user))-> ChatResponse:
    """
    Endpoint to handle query requests. It takes a question and optional flags,
    processes the request using the RAG service, and returns a chat response.
    """
    logger.info(f"Received query request: {body.question} with flags: top_k={body.top_k}, search_mode={body.search_mode}, enable_rerank={body.enable_rerank}")
    return await run_rag_async(
               body.question,   
                   flags=
                   {
                   "top_k": body.top_k  if body.top_k is not None else None,
                   "search_mode": body.search_mode if body.search_mode is not None else None,
                   "rerank": body.enable_rerank if body.enable_rerank is not None else None,
                   "hyde": body.enable_hyde if body.enable_hyde is not None else None,
                   }
    )