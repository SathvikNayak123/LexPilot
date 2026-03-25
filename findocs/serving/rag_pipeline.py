"""Full RAG pipeline: query -> retrieve -> generate.

Orchestrates query routing, hybrid search with reranking, prompt versioning
via Langfuse, and two-tier model generation (fine-tuned primary with GPT-4o
fallback).  Every request is traced end-to-end in Langfuse so that retrieval
quality, latency, and generation faithfulness can be monitored per-query.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import structlog
from langfuse import Langfuse
from pydantic import BaseModel, Field

from findocs.serving.fallback_chain import FallbackChain
from findocs.serving.model_server import ModelServer
from findocs.serving.prompt_registry import PromptRegistry

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol definitions for pluggable dependencies
# ---------------------------------------------------------------------------


class QueryRouteConfig(BaseModel):
    """Configuration returned by the query router."""

    query_type: str = Field(description="Classified query type (e.g. 'factual', 'analytical', 'comparative').")
    doc_types: list[str] = Field(default_factory=list, description="Document types to search over.")
    top_k: int = Field(default=5, description="Number of results to retrieve.")
    require_rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking.")


class RetrievedContext(BaseModel):
    """A single chunk retrieved by hybrid search."""

    content: str = Field(description="Text content of the retrieved chunk.")
    doc_title: str = Field(default="", description="Title of the source document.")
    doc_type: str = Field(default="", description="Document type label.")
    doc_date: datetime | None = Field(default=None, description="Publication date of the source document.")
    url: str | None = Field(default=None, description="Source URL.")
    page_num: int = Field(default=1, description="Page number in the source document.")
    relevance_score: float = Field(default=0.0, description="Relevance score after reranking.")


@runtime_checkable
class QueryRouter(Protocol):
    """Protocol for query routing implementations."""

    async def classify(self, question: str) -> QueryRouteConfig:
        """Classify an incoming question and return routing configuration."""
        ...


@runtime_checkable
class HybridSearcher(Protocol):
    """Protocol for hybrid search implementations."""

    async def search(
        self,
        query: str,
        top_k: int = 5,
        doc_types: list[str] | None = None,
        rerank: bool = True,
    ) -> list[RetrievedContext]:
        """Run hybrid search (dense + sparse) with optional reranking."""
        ...


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SourceCitation(BaseModel):
    """A single source citation attached to a RAG response."""

    doc_title: str = Field(description="Title of the cited document.")
    doc_type: str = Field(description="Document type (e.g. 'rbi_circular').")
    doc_date: datetime | None = Field(default=None, description="Publication date, if available.")
    url: str | None = Field(default=None, description="URL to the original document.")
    page_num: int = Field(default=1, description="Page number of the cited content.")
    relevance_score: float = Field(default=0.0, description="Relevance score from the reranker.")


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline."""

    answer: str = Field(description="Generated answer text.")
    sources: list[SourceCitation] = Field(default_factory=list, description="Cited source documents.")
    confidence_score: float = Field(description="Pipeline confidence in the answer (0.0-1.0).")
    query_type: str = Field(description="Classified query type.")
    retrieval_latency_ms: float = Field(description="Time spent on retrieval in milliseconds.")
    generation_latency_ms: float = Field(description="Time spent on generation in milliseconds.")
    model_used: str = Field(description="Tag indicating which model served the generation ('primary' or 'fallback').")
    langfuse_trace_id: str = Field(description="Langfuse trace ID for this request.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end RAG pipeline: route -> retrieve -> prompt -> generate.

    Integrates query routing, hybrid search with reranking, versioned prompts
    from Langfuse, two-tier generation (fine-tuned + GPT-4o fallback), and
    full Langfuse tracing.

    Parameters
    ----------
    searcher:
        A hybrid searcher that executes dense + sparse retrieval and
        optional cross-encoder reranking.
    router:
        A query router that classifies the incoming question.
    prompt_registry:
        Langfuse-backed prompt registry for versioned templates.
    fallback_chain:
        Two-tier model generation chain (primary -> fallback).
    langfuse:
        Langfuse client for tracing.
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        router: QueryRouter,
        prompt_registry: PromptRegistry,
        fallback_chain: FallbackChain,
        langfuse: Langfuse,
    ) -> None:
        self._searcher: HybridSearcher = searcher
        self._router: QueryRouter = router
        self._prompt_registry: PromptRegistry = prompt_registry
        self._fallback_chain: FallbackChain = fallback_chain
        self._langfuse: Langfuse = langfuse

        logger.info("rag_pipeline.init")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(
        self,
        question: str,
        session_id: str | None = None,
    ) -> RAGResponse:
        """Execute the full RAG pipeline for a user question.

        Steps
        -----
        1. Router classifies the query into a ``QueryRouteConfig``.
        2. Hybrid search retrieves relevant chunks.
        3. A versioned prompt template is fetched from Langfuse and compiled.
        4. The fallback chain generates an answer (fine-tuned first, GPT-4o
           on failure).
        5. A confidence score is computed from retrieval scores and answer
           characteristics.
        6. The full trace is logged to Langfuse.
        7. A ``RAGResponse`` is returned.

        Parameters
        ----------
        question:
            The user's natural-language question.
        session_id:
            Optional session identifier for multi-turn tracking.

        Returns
        -------
        RAGResponse
            The generated answer with sources, metrics, and trace ID.
        """

        # --- Start Langfuse trace -----------------------------------------
        trace = self._langfuse.trace(
            name="rag_pipeline.query",
            input={"question": question},
            session_id=session_id,
        )
        trace_id: str = trace.id

        log = logger.bind(trace_id=trace_id, session_id=session_id)
        log.info("rag_pipeline.query.start", question=question[:120])

        # --- Step 1: Route the query --------------------------------------
        route_span = trace.span(name="query_routing", input={"question": question})
        try:
            route_config: QueryRouteConfig = await self._router.classify(question)
        except Exception as exc:
            log.warning("rag_pipeline.router_error", error=str(exc))
            route_config = QueryRouteConfig(
                query_type="general",
                doc_types=[],
                top_k=5,
                require_rerank=True,
            )
        route_span.end(output=route_config.model_dump())

        log.info(
            "rag_pipeline.query.routed",
            query_type=route_config.query_type,
            doc_types=route_config.doc_types,
            top_k=route_config.top_k,
        )

        # --- Step 2: Hybrid search ----------------------------------------
        retrieval_span = trace.span(
            name="hybrid_search",
            input={
                "question": question,
                "top_k": route_config.top_k,
                "doc_types": route_config.doc_types,
            },
        )
        retrieval_start = time.perf_counter()

        contexts: list[RetrievedContext] = await self._searcher.search(
            query=question,
            top_k=route_config.top_k,
            doc_types=route_config.doc_types if route_config.doc_types else None,
            rerank=route_config.require_rerank,
        )

        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
        retrieval_span.end(
            output={
                "num_results": len(contexts),
                "latency_ms": round(retrieval_latency_ms, 2),
            },
        )

        log.info(
            "rag_pipeline.query.retrieved",
            num_contexts=len(contexts),
            latency_ms=round(retrieval_latency_ms, 2),
        )

        # --- Step 3: Build prompt from versioned template -----------------
        prompt_span = trace.span(name="prompt_construction")

        contexts_text = self._format_contexts(contexts)
        doc_types_available = ", ".join(sorted({c.doc_type for c in contexts if c.doc_type})) or "general"
        current_date = datetime.now().strftime("%Y-%m-%d")

        template_variables: dict[str, str] = {
            "question": question,
            "contexts": contexts_text,
            "doc_types_available": doc_types_available,
            "current_date": current_date,
        }

        system_prompt = await self._prompt_registry.get_prompt(
            "rag_system",
            variables=template_variables,
        )
        user_prompt = await self._prompt_registry.get_prompt(
            "rag_user",
            variables=template_variables,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_span.end(
            output={
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
                "num_contexts_in_prompt": len(contexts),
            },
        )

        # --- Step 4: Generate answer --------------------------------------
        generation_span = trace.span(
            name="generation",
            input={"messages_count": len(messages)},
        )
        generation_start = time.perf_counter()

        generation_result, model_used = await self._fallback_chain.generate(
            messages,
            max_tokens=1024,
        )

        generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0
        generation_span.end(
            output={
                "model_used": model_used,
                "model": generation_result.model,
                "tokens_used": generation_result.tokens_used,
                "latency_ms": round(generation_latency_ms, 2),
            },
        )

        log.info(
            "rag_pipeline.query.generated",
            model_used=model_used,
            tokens_used=generation_result.tokens_used,
            latency_ms=round(generation_latency_ms, 2),
        )

        # --- Step 5: Compute confidence score -----------------------------
        confidence_score = self._compute_confidence(
            contexts=contexts,
            answer=generation_result.content,
            model_used=model_used,
        )

        # --- Step 6: Build source citations -------------------------------
        sources: list[SourceCitation] = [
            SourceCitation(
                doc_title=ctx.doc_title,
                doc_type=ctx.doc_type,
                doc_date=ctx.doc_date,
                url=ctx.url,
                page_num=ctx.page_num,
                relevance_score=round(ctx.relevance_score, 4),
            )
            for ctx in contexts
        ]

        # --- Step 7: Finalise Langfuse trace ------------------------------
        response = RAGResponse(
            answer=generation_result.content,
            sources=sources,
            confidence_score=round(confidence_score, 4),
            query_type=route_config.query_type,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            model_used=model_used,
            langfuse_trace_id=trace_id,
        )

        trace.update(
            output=response.model_dump(mode="json"),
            metadata={
                "confidence_score": response.confidence_score,
                "model_used": model_used,
                "num_sources": len(sources),
                "query_type": route_config.query_type,
            },
        )

        # Score the trace for later analysis
        trace.score(name="confidence", value=confidence_score)

        log.info(
            "rag_pipeline.query.done",
            confidence=response.confidence_score,
            num_sources=len(sources),
            total_latency_ms=round(retrieval_latency_ms + generation_latency_ms, 2),
        )

        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_contexts(contexts: list[RetrievedContext]) -> str:
        """Format retrieved contexts into a numbered block for the prompt.

        Parameters
        ----------
        contexts:
            Ordered list of retrieved context chunks.

        Returns
        -------
        str
            A formatted string with one section per context, including
            metadata headers and content.
        """

        if not contexts:
            return "(No relevant documents found.)"

        parts: list[str] = []
        for idx, ctx in enumerate(contexts, start=1):
            header_items: list[str] = []
            if ctx.doc_title:
                header_items.append(f"Title: {ctx.doc_title}")
            if ctx.doc_type:
                header_items.append(f"Type: {ctx.doc_type}")
            if ctx.doc_date:
                header_items.append(f"Date: {ctx.doc_date.strftime('%Y-%m-%d')}")
            if ctx.page_num:
                header_items.append(f"Page: {ctx.page_num}")

            header = " | ".join(header_items) if header_items else "Unknown source"

            parts.append(
                f"[Source {idx}] ({header})\n{ctx.content}"
            )

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _compute_confidence(
        contexts: list[RetrievedContext],
        answer: str,
        model_used: str,
    ) -> float:
        """Compute a heuristic confidence score for the generated answer.

        The score combines:

        * **Retrieval quality** (40%): mean relevance score of the top
          contexts, normalised to [0, 1].
        * **Context coverage** (30%): number of contexts retrieved relative
          to a target of 5, capped at 1.0.
        * **Answer substance** (15%): penalises very short answers that are
          likely uninformative.
        * **Model tier** (15%): a slight boost for the fine-tuned primary
          model vs. the generic fallback.

        Parameters
        ----------
        contexts:
            Retrieved context chunks (with relevance scores).
        answer:
            The generated answer text.
        model_used:
            ``"primary"`` or ``"fallback"``.

        Returns
        -------
        float
            Confidence score in [0.0, 1.0].
        """

        # --- Retrieval quality (0-1) --------------------------------------
        if contexts:
            scores = [c.relevance_score for c in contexts]
            # Clamp individual scores to [0, 1] before averaging
            clamped = [max(0.0, min(1.0, s)) for s in scores]
            retrieval_quality = sum(clamped) / len(clamped)
        else:
            retrieval_quality = 0.0

        # --- Context coverage (0-1) ---------------------------------------
        target_contexts = 5
        coverage = min(len(contexts) / target_contexts, 1.0) if target_contexts > 0 else 0.0

        # --- Answer substance (0-1) ---------------------------------------
        word_count = len(answer.split())
        if word_count < 10:
            substance = 0.3
        elif word_count < 30:
            substance = 0.6
        elif word_count < 80:
            substance = 0.85
        else:
            substance = 1.0

        # --- Model tier (0-1) ---------------------------------------------
        model_score = 0.9 if model_used == "primary" else 0.7

        # --- Weighted combination -----------------------------------------
        confidence = (
            0.40 * retrieval_quality
            + 0.30 * coverage
            + 0.15 * substance
            + 0.15 * model_score
        )

        return max(0.0, min(1.0, confidence))
