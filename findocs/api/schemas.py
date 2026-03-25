"""Shared Pydantic schemas for the FinDocs API.

Defines request and response models used across all API routes.  Every model
carries full type hints and docstrings to support automatic OpenAPI schema
generation by FastAPI.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query endpoint schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the ``POST /query`` endpoint.

    Attributes:
        question: The natural-language question to answer.
        session_id: Optional conversation session identifier for
            multi-turn context.  When ``None``, a new session is created.
        doc_type_filter: Optional filter to restrict retrieval to a
            specific document type (e.g. ``"rbi_circular"``).
        top_k: Number of context chunks to retrieve.  Defaults to 5.
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


class SourceChunk(BaseModel):
    """A single source chunk returned alongside the answer.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        content: Text content of the chunk.
        doc_type: Document type label.
        doc_date: ISO-format publication date of the source document.
        page_num: Page number in the original PDF.
        score: Relevance score from the retrieval pipeline.
    """

    chunk_id: str = Field(description="Unique identifier of the chunk.")
    content: str = Field(description="Text content of the chunk.")
    doc_type: str | None = Field(default=None, description="Document type label.")
    doc_date: str | None = Field(
        default=None,
        description="Publication date of the source document (ISO format).",
    )
    page_num: int | None = Field(
        default=None,
        description="Page number in the original PDF.",
    )
    score: float = Field(description="Relevance score from retrieval/reranking.")


class QueryResponse(BaseModel):
    """Response body for the ``POST /query`` endpoint.

    Wraps the RAG pipeline output with additional API-level metadata such
    as a unique request identifier and timing information.

    Attributes:
        request_id: Unique identifier for this request (UUIDv4).
        question: Echo of the original question for client convenience.
        answer: The generated answer text.
        sources: List of source chunks used as context for generation.
        session_id: Session identifier (echoed back or newly created).
        model_used: Identifier of the LLM that generated the answer.
        latency_ms: End-to-end response latency in milliseconds.
        trace_id: Langfuse trace identifier for observability.
        created_at: ISO-format timestamp of when the response was created.
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this API request.",
    )
    question: str = Field(description="Echo of the original question.")
    answer: str = Field(description="Generated answer text.")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Source chunks used as context for generation.",
    )
    session_id: str = Field(description="Session identifier.")
    model_used: str = Field(
        default="",
        description="Identifier of the LLM used for generation.",
    )
    latency_ms: float = Field(
        default=0.0,
        description="End-to-end response latency in milliseconds.",
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
        score: User rating.  ``1`` for positive, ``-1`` for negative.
        comment: Optional free-text comment from the user.
    """

    trace_id: str = Field(
        ...,
        min_length=1,
        description="Langfuse trace ID of the query being rated.",
    )
    score: Literal[1, -1] = Field(
        ...,
        description="User rating: 1 (positive) or -1 (negative).",
    )
    comment: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional free-text feedback comment.",
    )


class FeedbackResponse(BaseModel):
    """Response body for the ``POST /feedback`` endpoint.

    Attributes:
        status: Outcome status string (``"accepted"``).
        trace_id: Echo of the trace ID that was rated.
        message: Human-readable confirmation message.
    """

    status: str = Field(
        default="accepted",
        description="Outcome status.",
    )
    trace_id: str = Field(description="Echo of the rated trace ID.")
    message: str = Field(
        default="Feedback recorded successfully.",
        description="Human-readable confirmation.",
    )


# ---------------------------------------------------------------------------
# Health endpoint schema
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint.

    Attributes:
        status: Overall health status (``"healthy"`` or ``"degraded"``).
        version: Application version string.
        qdrant_connected: Whether the Qdrant vector store is reachable.
        postgres_connected: Whether Postgres is reachable.
        timestamp: UTC timestamp of the health check.
    """

    status: str = Field(
        default="healthy",
        description="Overall service health status.",
    )
    version: str = Field(
        default="0.1.0",
        description="Application version.",
    )
    qdrant_connected: bool = Field(
        default=False,
        description="Whether the Qdrant vector store is reachable.",
    )
    postgres_connected: bool = Field(
        default=False,
        description="Whether the Postgres database is reachable.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the health check.",
    )
