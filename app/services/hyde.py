import asyncio
import logging
import re
import numpy as np

from app.config import settings
from app.models import RetrievedChunk
from app.services.embedding_service import embed_texts
from app.services.llm_service import generate_async
from app.services.vector_store import search

logger = logging.getLogger(__name__)

_HYDE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given a user question, write a brief, plausible answer "
    "(2-3 sentences) that would help retrieve relevant documents. Write only the answer, "
    "no preamble."
)


def _normalize_text(text: str) -> str:
    """Normalize whitespace for deduplication."""
    return re.sub(r"\s+", " ", text.strip())


class HyDERetriever:
    def __init__(self, num_hypotheses: int | None = None) -> None:
        self.num_hypotheses = num_hypotheses or settings.hyde_num_hypotheses

    async def _generate_single_hypothesis(self, question: str) -> str | None:
        """Helper to generate a single hypothesis with localized exception handling."""
        try:
            response = await generate_async(
                system_prompt=_HYDE_SYSTEM_PROMPT,
                user_message=question,
                model=settings.llm_model_answer,
                temperature=0.7,
            )
            hypothesis = response.get("text", "").strip()
            return hypothesis if hypothesis else None
        except Exception as e:
            logger.warning(f"Failed to generate hypothesis: {e}")
            return None

    async def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Async retrieval using parallel HyDE hypothesis generation and vector averaging."""
        if not question or not question.strip():
            return []

        # 1. Generate all hypotheses in parallel
        tasks = [
            self._generate_single_hypothesis(question)
            for _ in range(self.num_hypotheses)
        ]
        results = await asyncio.gather(*tasks)

        # 2. Extract valid hypotheses
        hypotheses = [h for h in results if h is not None]

        # 3. Combine hypotheses with the original question as context
        # (If all hypothesis calls fail, falls back seamlessly to question-only)
        all_texts = hypotheses + [question]

        # 4. Batch embed all texts
        try:
            embeddings = embed_texts(all_texts)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

        if not embeddings:
            return []

        # 5. Average the vectors (Standard HyDE technique)
        mean_embedding = np.mean(embeddings, axis=0).tolist()

        # 6. Perform a single database lookup
        try:
            results = search(mean_embedding, top_k=top_k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        # 7. Deduplicate chunks by text content
        deduped: dict[str, RetrievedChunk] = {}
        for chunk in results:
            key = _normalize_text(chunk.text)
            if key not in deduped or chunk.score > deduped[key].score:
                deduped[key] = chunk

        merged = sorted(deduped.values(), key=lambda c: c.score, reverse=True)
        return merged[:top_k]