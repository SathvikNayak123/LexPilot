"""Shared Pydantic schemas for the FinDocs API.

Defines request and response models used across all API routes.  Every model
carries full type hints and docstrings to support automatic OpenAPI schema
generation by FastAPI.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query endpoint schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the ``POST /query`` endpoint.

    Attributes:
        question: The natural-language question to answer.
        session_id: Optional conversation session identifier.
        doc_type_filter: Optional filter to restrict retrieval to a document type.
        top_k: Number of context chunks to retrieve.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The natural-language question to answer.",
        examples=["What are the latest RBI guidelines on digital lending?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversations.",
    )
    doc_type_filter: str | None = Field(
        default=None,
        description=(
            "Restrict retrieval to a specific document type. "
            "Accepted values: rbi_circular, sebi_factsheet, nse_annual_report."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve.",
    )


class SourceCitationSchema(BaseModel):
    """A single source citation returned alongside the answer.

    Attributes:
        doc_title: Title of the source document.
        doc_type: Document type label.
        doc_date: Publication date of the source document.
        url: URL of the source document.
        page_num: Page number in the original PDF.
        relevance_score: Relevance score from the retrieval pipeline.
    """

    doc_title: str = Field(description="Title of the source document.")
    doc_type: str = Field(description="Document type label.")
    doc_date: datetime | None = Field(default=None, description="Publication date.")
    url: str | None = Field(default=None, description="URL of the source document.")
    page_num: int = Field(description="Page number in the original PDF.")
    relevance_score: float = Field(description="Relevance score from retrieval/reranking.")


class QueryResponse(BaseModel):
    """Response body for the ``POST /query`` endpoint.

    Attributes:
        request_id: Unique identifier for this request (UUIDv4).
        answer: The generated answer text.
        sources: List of source citations used as context for generation.
        confidence_score: Pipeline confidence in the answer.
        query_type: Classified query type from the router.
        retrieval_latency_ms: Retrieval stage latency in milliseconds.
        generation_latency_ms: Generation stage latency in milliseconds.
        model_used: Identifier of the LLM that generated the answer.
        trace_id: Langfuse trace identifier for observability.
        created_at: UTC timestamp of when the response was created.
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this API request.",
    )
    answer: str = Field(description="Generated answer text.")
    sources: list[SourceCitationSchema] = Field(
        default_factory=list,
        description="Source citations used as context for generation.",
    )
    confidence_score: float = Field(
        default=0.0,
        description="Pipeline confidence score (0.0 to 1.0).",
    )
    query_type: str = Field(
        default="factual",
        description="Classified query type.",
    )
    retrieval_latency_ms: float = Field(
        default=0.0,
        description="Retrieval stage latency in milliseconds.",
    )
    generation_latency_ms: float = Field(
        default=0.0,
        description="Generation stage latency in milliseconds.",
    )
    model_used: str = Field(
        default="",
        description="Identifier of the LLM used for generation.",
    )
    trace_id: str = Field(
        default="",
        description="Langfuse trace ID for observability.",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of response creation.",
    )


# ---------------------------------------------------------------------------
# Feedback endpoint schemas
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """Request body for the ``POST /feedback`` endpoint.

    Attributes:
        trace_id: The Langfuse trace ID of the query being rated.
        session_id: Optional session ID.
        question: The original question.
        answer: The generated answer.
        score: User rating: 1 (positive), 0 (neutral), -1 (negative).
        comment: Optional free-text comment from the user.
    """

    trace_id: str = Field(
        ...,
        min_length=1,
        description="Langfuse trace ID of the query being rated.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID.",
    )
    question: str = Field(
        default="",
        description="The original question.",
    )
    answer: str = Field(
        default="",
        description="The generated answer.",
    )
    score: int = Field(
        ...,
        ge=-1,
        le=1,
        description="User rating: 1 (positive), 0 (neutral), -1 (negative).",
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional free-text feedback comment.",
    )


class FeedbackResponse(BaseModel):
    """Response body for the ``POST /feedback`` endpoint.

    Attributes:
        feedback_id: ID of the stored feedback record.
        status: Outcome status string.
    """

    feedback_id: str = Field(description="ID of the stored feedback record.")
    status: str = Field(
        default="stored",
        description="Outcome status.",
    )


# ---------------------------------------------------------------------------
# Health endpoint schema
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint.

    Attributes:
        status: Overall health status.
        version: Application version string.
    """

    status: str = Field(default="ok", description="Overall service health status.")
    version: str = Field(default="0.1.0", description="Application version.")
