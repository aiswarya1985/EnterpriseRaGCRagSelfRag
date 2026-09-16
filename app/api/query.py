from fastapi import APIRouter, Depends
from app.middleware.auth import User, get_current_user
from app.models import ChatResponse, QueryRequest,PendingSQLBlock
from app.services.rag_service import run_rag_async
from loguru import logger
import uuid 
from langgraph.types import Command
from pydantic import BaseModel
from app.core.graph import graph

router=APIRouter(tags=["query"])

class SqlExecuteRequest(BaseModel):
    query_id: str
    approved: bool


@router.post("/query", response_model=ChatResponse)
async def query(
    body: QueryRequest, 
    user: User = Depends(get_current_user))-> ChatResponse:
    """
    Endpoint to handle query requests. It takes a question and optional flags,
    processes the request using the RAG service, and returns a chat response.
    """
    try:
        logger.info(f"Received query request: {body.question} with flags: top_k={body.top_k}, search_mode={body.search_mode}, enable_rerank={body.enable_rerank}")
        thread_id = str(uuid.uuid4())
        config={"configurable": {"thread_id": thread_id}}

        result = graph.invoke(
        {
            "question": body.question,
            "user_id": body.user_id,
            "flags":body.model_dump()

        },
        config=config
        )
        if "__interrupt__" in result:
            intr=result["__interrupt__"][0].value
            return ChatResponse(
                answer="",
                sources=[],
                confidence=0.0,
                pending_sql=PendingSQLBlock(
                    sql=intr["sql"],
                    query_id=thread_id,
                    explanation=intr.get("explanation", "")

                )
            )
        return ChatResponse(
        answer=result.get("final_answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0)
        )
    except Exception:
        logger.exception("exception occured in the query processing:")

@router.post("/query/sql/execute", response_model=ChatResponse)
async def execute_sql(
    body: SqlExecuteRequest,
    user: User = Depends(get_current_user),
) -> ChatResponse:
    
    config = {"configurable": {"thread_id": body.query_id}}

    result = graph.invoke(
        Command(resume={"approved": body.approved}),
        config=config,
    )

    return ChatResponse(
        answer=result.get("final_answer", "SQL query was not approved."),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        cache_hit=result.get("cache_hit", False),
        metadata=result.get("metadata", {}),
    )


    # to comment control+/     
    # return await run_rag_async(
    #            body.question,   
    #                flags=
    #                {
    #                "top_k": body.top_k  if body.top_k is not None else None,
    #                "search_mode": body.search_mode if body.search_mode is not None else None,
    #                "rerank": body.enable_rerank if body.enable_rerank is not None else None,
    #                "hyde": body.enable_hyde if body.enable_hyde is not None else None,
    #                "crag": body.enable_crag if body.enable_crag is not None else None,    
    #                "srag": body.enable_self_reflective if body.enable_self_reflective is not None else None
    #                }
    #)