"""POST /query – main RAG endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from findocs.api.schemas import QueryRequest, QueryResponse, SourceCitationSchema
from findocs.config.config import get_settings
from findocs.monitoring.langfuse_monitor import LangfuseMonitor
from findocs.processing.embedder import EmbeddingService
from findocs.retrieval.hybrid_search import HybridSearcher
from findocs.retrieval.qdrant_client import FinDocsQdrantClient
from findocs.retrieval.query_router import QueryRouter
from findocs.retrieval.reranker import Reranker
from findocs.serving.fallback_chain import FallbackChain
from findocs.serving.model_server import ModelServer
from findocs.serving.prompt_registry import PromptRegistry
from findocs.serving.rag_pipeline import RAGPipeline

logger: structlog.stdlib.BoundLogger = structlog.get_logger()
settings = get_settings()

router = APIRouter()

# Lazy singletons — initialised on first request
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """Build or return the cached RAG pipeline singleton."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is not None:
        return _pipeline

    embedder = EmbeddingService()
    qdrant = FinDocsQdrantClient(url=settings.QDRANT_URL, collection_name=settings.QDRANT_COLLECTION)
    reranker = Reranker()
    searcher = HybridSearcher(qdrant_client=qdrant, reranker=reranker, embedder=embedder)
    router_instance = QueryRouter()
    prompt_registry = PromptRegistry()
    model_server = ModelServer()
    fallback_chain = FallbackChain(model_server=model_server)
    monitor = LangfuseMonitor()

    _pipeline = RAGPipeline(
        searcher=searcher,
        router=router_instance,
        prompt_registry=prompt_registry,
        model_server=model_server,
        fallback_chain=fallback_chain,
        monitor=monitor,
    )
    return _pipeline


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run a question through the full RAG pipeline.

    Args:
        request: The query request containing the user's question.

    Returns:
        RAG-generated answer with sources and metadata.
    """
    pipeline = _get_pipeline()

    try:
        result = await pipeline.query(
            question=request.question,
            session_id=request.session_id,
            doc_type_filter=request.doc_type_filter,
        )
    except Exception as exc:
        logger.error("rag_pipeline_error", error=str(exc), question=request.question)
        raise HTTPException(status_code=500, detail="Internal error processing query") from exc

    sources = [
        SourceCitationSchema(
            doc_title=s.doc_title,
            doc_type=s.doc_type,
            doc_date=s.doc_date,
            url=s.url,
            page_num=s.page_num,
            relevance_score=s.relevance_score,
        )
        for s in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        confidence_score=result.confidence_score,
        query_type=result.query_type,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        model_used=result.model_used,
        trace_id=result.langfuse_trace_id,
    )
