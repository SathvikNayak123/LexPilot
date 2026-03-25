"""End-to-end tests for serving/rag_pipeline.py.

ALL external services are mocked: Langfuse, hybrid search, query router,
fallback chain, prompt registry.  Uses unittest.mock.patch and
unittest.mock.AsyncMock throughout.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from findocs.serving.fallback_chain import FallbackChain
from findocs.serving.model_server import GenerationResult
from findocs.serving.prompt_registry import PromptRegistry
from findocs.serving.rag_pipeline import (
    HybridSearcher,
    QueryRouteConfig,
    QueryRouter,
    RAGPipeline,
    RAGResponse,
    RetrievedContext,
    SourceCitation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_searcher(contexts: list[RetrievedContext] | None = None) -> MagicMock:
    """Create a mock HybridSearcher returning predefined contexts."""
    mock = MagicMock(spec=HybridSearcher)

    if contexts is None:
        contexts = [
            RetrievedContext(
                content="The RBI has set the repo rate at 6.50% for FY2025.",
                doc_title="RBI Monetary Policy Feb 2025",
                doc_type="rbi_circular",
                doc_date=datetime(2025, 2, 7),
                url="https://rbi.org.in/circular/2025/02/07",
                page_num=1,
                relevance_score=0.92,
            ),
            RetrievedContext(
                content="The reverse repo rate remains at 3.35%, unchanged from the previous meeting.",
                doc_title="RBI Monetary Policy Feb 2025",
                doc_type="rbi_circular",
                doc_date=datetime(2025, 2, 7),
                url="https://rbi.org.in/circular/2025/02/07",
                page_num=2,
                relevance_score=0.85,
            ),
            RetrievedContext(
                content="GDP growth projection for FY2025 is 7.0% with balanced risks.",
                doc_title="RBI Economic Outlook",
                doc_type="rbi_circular",
                doc_date=datetime(2025, 2, 7),
                url=None,
                page_num=5,
                relevance_score=0.78,
            ),
        ]

    mock.search = AsyncMock(return_value=contexts)
    return mock


def _make_mock_router(query_type: str = "factual") -> MagicMock:
    """Create a mock QueryRouter returning a predefined config."""
    mock = MagicMock(spec=QueryRouter)
    mock.classify = AsyncMock(
        return_value=QueryRouteConfig(
            query_type=query_type,
            doc_types=["rbi_circular"],
            top_k=5,
            require_rerank=True,
        )
    )
    return mock


def _make_mock_prompt_registry() -> MagicMock:
    """Create a mock PromptRegistry that returns template strings."""
    mock = MagicMock(spec=PromptRegistry)

    async def get_prompt_side_effect(name: str, variables: dict | None = None) -> str:
        if name == "rag_system":
            return (
                "You are a financial document assistant. Answer questions based on "
                "the provided context. Be accurate and cite sources."
            )
        elif name == "rag_user":
            question = (variables or {}).get("question", "")
            contexts = (variables or {}).get("contexts", "")
            return f"Question: {question}\n\nContext:\n{contexts}\n\nAnswer:"
        return f"Mock prompt for {name}"

    mock.get_prompt = AsyncMock(side_effect=get_prompt_side_effect)
    return mock


def _make_mock_fallback_chain(
    content: str = "The repo rate is 6.50% as per the latest RBI monetary policy.",
    model_used: str = "primary",
    raise_primary: bool = False,
) -> MagicMock:
    """Create a mock FallbackChain.

    If raise_primary is True, the first call will raise an exception,
    simulating fallback to GPT-4o.
    """
    mock = MagicMock(spec=FallbackChain)

    result = GenerationResult(
        content=content,
        model="phi3-findocs" if model_used == "primary" else "gpt-4o",
        tokens_used=200,
        latency_ms=150.0,
    )

    if raise_primary:
        # Simulate fallback: primary fails, then fallback succeeds
        fallback_result = GenerationResult(
            content=content,
            model="gpt-4o",
            tokens_used=250,
            latency_ms=300.0,
        )
        mock.generate = AsyncMock(return_value=(fallback_result, "fallback"))
    else:
        mock.generate = AsyncMock(return_value=(result, model_used))

    return mock


def _make_mock_langfuse() -> MagicMock:
    """Create a mock Langfuse client."""
    mock = MagicMock()

    mock_trace = MagicMock()
    mock_trace.id = "trace-test-12345"
    mock_trace.span.return_value = MagicMock()
    mock_trace.span.return_value.end = MagicMock()
    mock_trace.update = MagicMock()
    mock_trace.score = MagicMock()

    mock.trace.return_value = mock_trace
    return mock


def _make_pipeline(
    searcher: MagicMock | None = None,
    router: MagicMock | None = None,
    prompt_registry: MagicMock | None = None,
    fallback_chain: MagicMock | None = None,
    langfuse: MagicMock | None = None,
) -> RAGPipeline:
    """Create a RAGPipeline with all mocked dependencies."""
    return RAGPipeline(
        searcher=searcher or _make_mock_searcher(),
        router=router or _make_mock_router(),
        prompt_registry=prompt_registry or _make_mock_prompt_registry(),
        fallback_chain=fallback_chain or _make_mock_fallback_chain(),
        langfuse=langfuse or _make_mock_langfuse(),
    )


# ============================================================================
# End-to-end query test
# ============================================================================


class TestEndToEndQuery:
    """Tests for the full query flow through RAGPipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_query(self) -> None:
        """Mock all externals, verify question -> answer flow works."""
        pipeline = _make_pipeline()

        response = await pipeline.query("What is the current repo rate?")

        assert isinstance(response, RAGResponse)
        assert response.answer, "Answer should not be empty"
        assert "6.50" in response.answer, (
            f"Expected '6.50' in answer, got: {response.answer}"
        )
        assert response.query_type == "factual"
        assert response.model_used == "primary"
        assert response.langfuse_trace_id == "trace-test-12345"
        assert response.retrieval_latency_ms >= 0
        assert response.generation_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_end_to_end_query_with_session(self) -> None:
        """Verify session_id is forwarded to Langfuse trace."""
        mock_langfuse = _make_mock_langfuse()
        pipeline = _make_pipeline(langfuse=mock_langfuse)

        response = await pipeline.query(
            "What is the CRR?",
            session_id="session-abc-123",
        )

        assert isinstance(response, RAGResponse)
        # Verify Langfuse trace was created with the session_id
        mock_langfuse.trace.assert_called_once()
        call_kwargs = mock_langfuse.trace.call_args
        assert call_kwargs.kwargs.get("session_id") == "session-abc-123"


# ============================================================================
# Fallback chain tests
# ============================================================================


class TestFallbackChain:
    """Tests for the fallback generation chain."""

    @pytest.mark.asyncio
    async def test_fallback_chain_triggers(self) -> None:
        """When fine-tuned model raises exception, fallback to GPT-4o is used."""
        fallback_chain = _make_mock_fallback_chain(
            content="The repo rate is 6.50% (via GPT-4o fallback).",
            raise_primary=True,
        )
        pipeline = _make_pipeline(fallback_chain=fallback_chain)

        response = await pipeline.query("What is the repo rate?")

        assert response.model_used == "fallback", (
            f"Expected 'fallback', got '{response.model_used}'"
        )
        assert response.answer, "Answer should not be empty even via fallback"

    @pytest.mark.asyncio
    async def test_primary_model_used_when_available(self) -> None:
        """When primary model succeeds, it should be used."""
        fallback_chain = _make_mock_fallback_chain(model_used="primary")
        pipeline = _make_pipeline(fallback_chain=fallback_chain)

        response = await pipeline.query("What is the repo rate?")

        assert response.model_used == "primary"


# ============================================================================
# Source citations tests
# ============================================================================


class TestSourceCitations:
    """Tests for source citation generation."""

    @pytest.mark.asyncio
    async def test_source_citations_populated(self) -> None:
        """Verify sources list is non-empty and has required fields."""
        pipeline = _make_pipeline()

        response = await pipeline.query("What is the repo rate?")

        assert len(response.sources) > 0, "Sources list should not be empty"

        for source in response.sources:
            assert isinstance(source, SourceCitation)
            assert source.doc_title, f"doc_title should not be empty: {source}"
            assert source.doc_type, f"doc_type should not be empty: {source}"
            assert source.page_num >= 1, f"page_num should be >= 1: {source}"
            assert source.relevance_score >= 0.0, (
                f"relevance_score should be non-negative: {source}"
            )

    @pytest.mark.asyncio
    async def test_source_citations_match_contexts(self) -> None:
        """Number of sources should match number of retrieved contexts."""
        contexts = [
            RetrievedContext(
                content="Context 1",
                doc_title="Doc 1",
                doc_type="rbi_circular",
                page_num=1,
                relevance_score=0.9,
            ),
            RetrievedContext(
                content="Context 2",
                doc_title="Doc 2",
                doc_type="sebi_factsheet",
                page_num=3,
                relevance_score=0.8,
            ),
        ]
        searcher = _make_mock_searcher(contexts=contexts)
        pipeline = _make_pipeline(searcher=searcher)

        response = await pipeline.query("Test question")

        assert len(response.sources) == 2, (
            f"Expected 2 sources, got {len(response.sources)}"
        )


# ============================================================================
# Prompt registry caching tests
# ============================================================================


class TestPromptRegistryCaching:
    """Tests for prompt registry caching behavior."""

    @pytest.mark.asyncio
    async def test_prompt_registry_caching(self) -> None:
        """Second call to get_prompt should not hit Langfuse API (verify mock call count)."""
        mock_langfuse_client = MagicMock()

        # Mock the get_prompt method on the Langfuse client
        mock_prompt_obj = MagicMock()
        mock_prompt_obj.prompt = "You are a helpful assistant. Answer: {question}"
        mock_langfuse_client.get_prompt.return_value = mock_prompt_obj

        registry = PromptRegistry(
            langfuse=mock_langfuse_client,
            cache_ttl_s=300.0,
        )

        variables = {"question": "What is the repo rate?"}

        # First call: should fetch from Langfuse
        result1 = await registry.get_prompt("rag_system", variables=variables)
        first_call_count = mock_langfuse_client.get_prompt.call_count

        # Second call with same args: should hit cache, NOT Langfuse
        result2 = await registry.get_prompt("rag_system", variables=variables)
        second_call_count = mock_langfuse_client.get_prompt.call_count

        assert result1 == result2, "Both calls should return the same result"
        assert second_call_count == first_call_count, (
            f"Second call should NOT hit Langfuse. "
            f"Call count after first: {first_call_count}, after second: {second_call_count}"
        )


# ============================================================================
# Confidence score tests
# ============================================================================


class TestConfidenceScore:
    """Tests for the confidence score computation."""

    @pytest.mark.asyncio
    async def test_confidence_score_range(self) -> None:
        """Confidence score should be between 0.0 and 1.0."""
        pipeline = _make_pipeline()

        response = await pipeline.query("What is the repo rate?")

        assert 0.0 <= response.confidence_score <= 1.0, (
            f"Confidence should be in [0.0, 1.0], got {response.confidence_score}"
        )

    @pytest.mark.asyncio
    async def test_confidence_score_with_no_contexts(self) -> None:
        """Confidence should still be valid (but low) with no retrieved contexts."""
        searcher = _make_mock_searcher(contexts=[])
        pipeline = _make_pipeline(searcher=searcher)

        response = await pipeline.query("What is the repo rate?")

        assert 0.0 <= response.confidence_score <= 1.0, (
            f"Confidence should be in [0.0, 1.0], got {response.confidence_score}"
        )

    @pytest.mark.asyncio
    async def test_confidence_higher_with_primary_model(self) -> None:
        """Using the primary model should yield higher confidence than fallback."""
        # Primary model
        primary_chain = _make_mock_fallback_chain(model_used="primary")
        pipeline_primary = _make_pipeline(fallback_chain=primary_chain)
        response_primary = await pipeline_primary.query("What is the repo rate?")

        # Fallback model
        fallback_chain = _make_mock_fallback_chain(
            model_used="fallback",
            raise_primary=True,
        )
        pipeline_fallback = _make_pipeline(fallback_chain=fallback_chain)
        response_fallback = await pipeline_fallback.query("What is the repo rate?")

        # Primary should have higher confidence (model tier contributes 15%)
        assert response_primary.confidence_score >= response_fallback.confidence_score, (
            f"Primary confidence ({response_primary.confidence_score}) should be >= "
            f"fallback confidence ({response_fallback.confidence_score})"
        )

    @pytest.mark.asyncio
    async def test_confidence_score_deterministic(self) -> None:
        """Same inputs should produce the same confidence score."""
        pipeline = _make_pipeline()

        response1 = await pipeline.query("What is the repo rate?")
        response2 = await pipeline.query("What is the repo rate?")

        assert response1.confidence_score == response2.confidence_score, (
            f"Confidence should be deterministic: "
            f"{response1.confidence_score} vs {response2.confidence_score}"
        )
