from __future__ import annotations

from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings


METRICS = [
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
]


def _get_ragas_llm():
    """Create a Ragas-compatible LLM using app settings."""
    client = OpenAI(api_key=settings.openai_api_key)

    return llm_factory(
        settings.llm_model_grader,
        client=client,
    )


def _get_ragas_embeddings():
    """Create Ragas-compatible embeddings using app settings."""
    lc_emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return LangchainEmbeddingsWrapper(lc_emb)


def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "user_input": [row["question"] for row in rows],
            "response": [row["answer"] for row in rows],
            "retrieved_contexts": [row["contexts"] for row in rows],
            "reference": [row["ground_truth"] for row in rows],
        }
    )


def run(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    dataset = build_dataset(rows)

    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=_get_ragas_llm(),
        embeddings=_get_ragas_embeddings(),
        show_progress=False,
    )

    return result.to_pandas().to_dict(orient="records")