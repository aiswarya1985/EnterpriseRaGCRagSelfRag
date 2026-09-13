from app.config import settings
from app.models import RetrievedChunk
from loguru import logger
import tavily



def search_web(query: str, max_results: int = 5) -> list[RetrievedChunk]:

    logger.info("Entering search_web function")  
    if not settings.tavily_api_key:
        raise ValueError("Tavily API key not configured")

    try:      
        client = tavily.TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        results = response.get("results", [])
        logger.info(f"Web search returned {(results)} results for query: {query}")
        return [
            RetrievedChunk(
                text=result["content"],
                source=result["url"],
                score=result.get("score", 0.0),
            )
            for result in results
        ]
    except Exception:
        logger.exception("Tavily web search failed")
        return []