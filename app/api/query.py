from fastapi import APIRouter, Depends
from app.middleware.auth import User, get_current_user
from app.models import ChatResponse, QueryRequest
from app.services.rag_service import run_rag

router=APIRouter(tags=["query"])

@router.post("/query", response_model=ChatResponse)
async def query(
    body: QueryRequest, 
    user: User = Depends(get_current_user))-> ChatResponse:
    """
    Endpoint to handle query requests. It takes a question and optional flags,
    processes the request using the RAG service, and returns a chat response.
    """
    return run_rag(body.question,   
                   flags={"top_k": body.top_k} 
                   if body.top_k is not None else None)