import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator
'''
{
  "ChatRequest": {
    "message": "What is the capital of France?"
  },
  "QueryRequest": {
    "question": "How does hybrid search work in vector databases?",
    "enable_rerank": true,
    "top_k": 5,
    "enable_hyde": false,
    "search_mode": "hybrid",
    "enable_crag": true,
    "enable_self_reflective": true
  },
  "RetrievedChunk": {
    "text": "Docling is an open-source library that parses complex PDFs into Markdown using Apple Silicon GPU acceleration.",
    "source": "docling_overview.pdf",
    "score": 0.92
  },
  "RetrievedChunkPreview": {
    "text": "Docling is an open-source library that parses complex PDFs...",
    "source": "docling_overview.pdf",
    "score": 0.92
  },
  "ResponseMetadata": {
    "route": "rag",
    "retrieved_chunks": [
      {
        "text": "Docling is an open-source library that parses complex PDFs...",
        "source": "docling_overview.pdf",
        "score": 0.92
      }
    ],
    "cache_hit": false,
    "reflection_iterations": 1,
    "reflection_score": 0.88,
    "refined_question": "What core document processing features does Docling offer?"
  },
  "PendingSQLBlock": {
    "sql": "SELECT COUNT(*) FROM user_documents WHERE created_at > '2026-01-01';",
    "query_id": "sql_req_98234",
    "explanation": "Calculates the total number of documents uploaded since the start of 2026."
  },
  "ChatResponse": {
    "answer": "Docling is an open-source document processing tool designed to convert PDFs into structured formats like Markdown.",
    "sources": [
      "docling_overview.pdf"
    ],
    "confidence": 0.95,
    "pending_sql": null,
    "cache_hit": false,
    "cost_saved": "$0.002",
    "metadata": {
      "route": "rag",
      "retrieved_chunks": [],
      "cache_hit": false,
      "reflection_iterations": 0,
      "reflection_score": null,
      "refined_question": null
    }
  },
  "CRAGEvaluation": {
    "relevance_score": 0.95,
    "relevance_label": "CORRECT",
    "confidence": 0.98,
    "reasoning": "The retrieved document passage directly answers the user query about PDF layout parsing."
  },
  "ReflectionResult": {
    "reflection_score": 0.40,
    "needs_regeneration": true,
    "refined_question": "What are the core technical features of Docling for document extraction?",
    "reasoning": "The initial generated answer was too vague and missed key OCR detail."
  }
}
'''
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message to the AI assistant",
    )
    @field_validator("message")
    @classmethod
    def validate_message_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty or whitespace only")
        injection_patterns = [
            r"(?i)(ignore\s+previous|ignore\s+above|forget\s+your\s+instructions)",
            r"(?i)(system\s*prompt|reveal\s+your\s+instructions|show\s+your\s+prompt)",
            r"(?i)(you\s+are\s+now|new\s+instructions|override\s+previous)",
            r"(?i)(<\s*script|javascript:|on\w+\s*=)",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, v):
                raise ValueError("Message contains potentially malicious content")

        if re.match(r"^[\W_]+$", v):
            raise ValueError("Message must contain actual text content")

        return v


class RetrievedChunkPreview(BaseModel):
    text: str
    source: str
    score: float = 0.0

class ResponseMetadata(BaseModel):
    route: str = "rag"
    retrieved_chunks: list[RetrievedChunkPreview] = Field(default_factory=list)
    cache_hit: bool = False
    reflection_iterations: int = 0
    reflection_score: float | None = None
    refined_question: str | None = None


class PendingSQLBlock(BaseModel):
    sql: str
    query_id: str
    explanation: str = ""


class ChatResponse(BaseModel):
    answer: str = Field(..., min_length=0)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    pending_sql: PendingSQLBlock | None = None
    cache_hit: bool = False
    cost_saved: str = "$0.00"
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question",
    )
    enable_rerank: bool = False
    top_k: int = Field(default=5, ge=1, le=50)
    enable_hyde: bool = False
    search_mode: Literal["dense", "sparse", "hybrid"] = "dense"
    enable_crag: bool = True
    enable_self_reflective: bool = False

    @field_validator("question")
    @classmethod
    def validate_question_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty or whitespace only")

        injection_patterns = [
            r"(?i)(ignore\s+previous|ignore\s+above|forget\s+your\s+instructions)",
            r"(?i)(system\s*prompt|reveal\s+your\s+instructions|show\s+your\s+prompt)",
            r"(?i)(you\s+are\s+now|new\s+instructions|override\s+previous)",
            r"(?i)(<\s*script|javascript:|on\w+\s*=)",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, v):
                raise ValueError("Question contains potentially malicious content")

        if re.match(r"^[\W_]+$", v):
            raise ValueError("Question must contain actual text content")

        return v


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float = 0.0

class CRAGEvaluation(BaseModel):
    relevance_score: float = 0.0
    relevance_label: str = "" 
    confidence: float = 0.0
    reasoning: str = ""

class ReflectionResult(BaseModel):
    """Self-RAG reflection on a generated answer."""

    reflection_score: float = 0.0
    needs_regeneration: bool = False
    refined_question: str = ""
    reasoning: str = ""
