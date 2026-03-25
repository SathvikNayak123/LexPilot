"""Tests for retrieval modules: hybrid_search (RRF), query_router, reranker, qdrant_client.

All external services (Qdrant, Anthropic, sentence-transformers) are mocked.
"""

from __future__ import annotations

import json
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# We test the static RRF method and models from hybrid_search
from findocs.retrieval.hybrid_search import HybridSearcher, RetrievedContext


# ---------------------------------------------------------------------------
# Lightweight stand-in for SearchResult to avoid importing heavy modules
# ---------------------------------------------------------------------------


class _MockSearchResult:
    """Lightweight stand-in matching the SearchResult interface for RRF testing."""

    def __init__(self, chunk_id: str, content: str = "", score: float = 0.0, payload: dict | None = None):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.payload = payload or {}

    def model_copy(self, update: dict | None = None) -> "_MockSearchResult":
        new = _MockSearchResult(
            chunk_id=self.chunk_id,
            content=self.content,
            score=update.get("score", self.score) if update else self.score,
            payload=self.payload.copy(),
        )
        return new


# ============================================================================
# RRF Fusion Tests
# ============================================================================


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion in HybridSearcher."""

    def test_rrf_fusion_formula(self) -> None:
        """Verify RRF score = sum(1/(k + rank)) with known inputs.

        With k=60:
        - doc_A: rank 1 in dense only -> score = 1/(60+1) = 1/61
        - doc_B: rank 2 in dense only -> score = 1/(60+2) = 1/62
        - doc_C: rank 1 in sparse only -> score = 1/(60+1) = 1/61
        """
        dense_results = [
            _MockSearchResult(chunk_id="doc_A", score=0.95),
            _MockSearchResult(chunk_id="doc_B", score=0.88),
        ]
        sparse_results = [
            _MockSearchResult(chunk_id="doc_C", score=15.3),
        ]

        fused = HybridSearcher.reciprocal_rank_fusion(dense_results, sparse_results, k=60)

        # Convert to dict for easier lookup
        fused_dict = dict(fused)

        expected_A = 1.0 / (60 + 1)  # rank 1 in dense
        expected_B = 1.0 / (60 + 2)  # rank 2 in dense
        expected_C = 1.0 / (60 + 1)  # rank 1 in sparse

        assert abs(fused_dict["doc_A"] - expected_A) < 1e-10, (
            f"doc_A score: expected {expected_A}, got {fused_dict['doc_A']}"
        )
        assert abs(fused_dict["doc_B"] - expected_B) < 1e-10, (
            f"doc_B score: expected {expected_B}, got {fused_dict['doc_B']}"
        )
        assert abs(fused_dict["doc_C"] - expected_C) < 1e-10, (
            f"doc_C score: expected {expected_C}, got {fused_dict['doc_C']}"
        )

    def test_rrf_deduplication(self) -> None:
        """Chunks appearing in both lists should get combined RRF scores.

        With k=60:
        - doc_X: rank 1 in dense + rank 2 in sparse
          score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        - doc_Y: rank 2 in dense only
          score = 1/(60+2) = 1/62
        - doc_Z: rank 1 in sparse only
          score = 1/(60+1) = 1/61
        """
        dense_results = [
            _MockSearchResult(chunk_id="doc_X", score=0.92),
            _MockSearchResult(chunk_id="doc_Y", score=0.85),
        ]
        sparse_results = [
            _MockSearchResult(chunk_id="doc_Z", score=12.5),
            _MockSearchResult(chunk_id="doc_X", score=10.2),  # duplicate
        ]

        fused = HybridSearcher.reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        fused_dict = dict(fused)

        expected_X = 1.0 / (60 + 1) + 1.0 / (60 + 2)  # appears in both
        expected_Y = 1.0 / (60 + 2)  # only in dense
        expected_Z = 1.0 / (60 + 1)  # only in sparse

        assert abs(fused_dict["doc_X"] - expected_X) < 1e-10, (
            f"doc_X combined score: expected {expected_X}, got {fused_dict['doc_X']}"
        )
        assert abs(fused_dict["doc_Y"] - expected_Y) < 1e-10
        assert abs(fused_dict["doc_Z"] - expected_Z) < 1e-10

        # doc_X should be ranked first (highest combined score)
        assert fused[0][0] == "doc_X", (
            f"Expected doc_X to be first, got {fused[0][0]}"
        )

    def test_rrf_empty_lists(self) -> None:
        """RRF with empty lists should return an empty result."""
        fused = HybridSearcher.reciprocal_rank_fusion([], [], k=60)
        assert fused == []

    def test_rrf_single_list(self) -> None:
        """RRF with one empty list should still produce correct scores."""
        dense_results = [
            _MockSearchResult(chunk_id="doc_A", score=0.9),
            _MockSearchResult(chunk_id="doc_B", score=0.8),
        ]
        fused = HybridSearcher.reciprocal_rank_fusion(dense_results, [], k=60)
        fused_dict = dict(fused)

        assert abs(fused_dict["doc_A"] - 1.0 / 61) < 1e-10
        assert abs(fused_dict["doc_B"] - 1.0 / 62) < 1e-10


# ============================================================================
# Query Router Tests
# ============================================================================


class TestQueryRouter:
    """Tests for the QueryRouter class."""

    @pytest.mark.asyncio
    async def test_query_router_factual(self, mock_anthropic_client: MagicMock) -> None:
        """'What is the repo rate?' should be classified as 'factual'."""
        # Configure mock to return factual classification
        content_block = MagicMock()
        content_block.text = json.dumps({
            "query_type": "factual",
            "top_k_override": 5,
            "chunk_type_filter": "child",
            "doc_type_filter": "rbi_circular",
            "run_multi_search": False,
            "sub_queries": [],
        })
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("findocs.retrieval.query_router.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(ANTHROPIC_API_KEY="test-key")

            with patch(
                "findocs.retrieval.query_router.anthropic.AsyncAnthropic",
                return_value=mock_anthropic_client,
            ):
                from findocs.retrieval.query_router import QueryRouter

                router = QueryRouter(anthropic_api_key="test-key")
                router._client = mock_anthropic_client

                config = await router.classify_and_route("What is the repo rate?")

        assert config.query_type == "factual", (
            f"Expected 'factual', got '{config.query_type}'"
        )

    @pytest.mark.asyncio
    async def test_query_router_numerical(self, mock_anthropic_client: MagicMock) -> None:
        """'What was the NAV of SBI Bluechip?' should be classified as 'numerical'."""
        content_block = MagicMock()
        content_block.text = json.dumps({
            "query_type": "numerical",
            "top_k_override": 5,
            "chunk_type_filter": "child",
            "doc_type_filter": "sebi_factsheet",
            "run_multi_search": False,
            "sub_queries": [],
        })
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("findocs.retrieval.query_router.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(ANTHROPIC_API_KEY="test-key")

            with patch(
                "findocs.retrieval.query_router.anthropic.AsyncAnthropic",
                return_value=mock_anthropic_client,
            ):
                from findocs.retrieval.query_router import QueryRouter

                router = QueryRouter(anthropic_api_key="test-key")
                router._client = mock_anthropic_client

                config = await router.classify_and_route(
                    "What was the NAV of SBI Bluechip Fund as of December 2025?"
                )

        assert config.query_type == "numerical", (
            f"Expected 'numerical', got '{config.query_type}'"
        )


# ============================================================================
# Reranker Tests
# ============================================================================


class TestReranker:
    """Tests for the Reranker class."""

    def test_reranker_ordering(self) -> None:
        """Higher relevance chunks should get higher scores after reranking."""

        # Mock the CrossEncoder
        with patch("findocs.retrieval.reranker.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            with patch("findocs.retrieval.reranker.CrossEncoder") as MockCrossEncoder:
                mock_cross_encoder = MagicMock()
                # Return scores where the second result is most relevant
                mock_cross_encoder.predict.return_value = np.array([0.3, 0.95, 0.1, 0.7])
                MockCrossEncoder.return_value = mock_cross_encoder

                from findocs.retrieval.reranker import Reranker
                from findocs.retrieval.qdrant_client import SearchResult

                reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

                results = [
                    SearchResult(chunk_id="c1", content="Repo rate info", score=0.8, payload={}),
                    SearchResult(chunk_id="c2", content="Detailed RBI policy on repo rate", score=0.75, payload={}),
                    SearchResult(chunk_id="c3", content="Weather forecast", score=0.6, payload={}),
                    SearchResult(chunk_id="c4", content="RBI monetary policy details", score=0.7, payload={}),
                ]

                reranked = reranker.rerank("What is the repo rate?", results, top_k=3)

        assert len(reranked) == 3, f"Expected 3 results, got {len(reranked)}"
        # The result with highest cross-encoder score (0.95) should be first
        assert reranked[0].chunk_id == "c2", (
            f"Expected c2 first (score 0.95), got {reranked[0].chunk_id}"
        )
        # Second should be c4 (score 0.7)
        assert reranked[1].chunk_id == "c4", (
            f"Expected c4 second (score 0.7), got {reranked[1].chunk_id}"
        )
        # Verify scores are in descending order
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True), (
            f"Scores should be in descending order, got {scores}"
        )


# ============================================================================
# Dense Search Tests (mocked Qdrant)
# ============================================================================


class TestDenseSearch:
    """Tests for dense search with mocked Qdrant client."""

    @pytest.mark.asyncio
    async def test_dense_search_returns_ordered(self) -> None:
        """Mock Qdrant, verify results are sorted by score descending."""

        with patch("findocs.retrieval.qdrant_client.get_settings") as mock_settings, \
             patch("findocs.retrieval.qdrant_client.AsyncQdrantClient") as MockQdrant, \
             patch("findocs.retrieval.qdrant_client.create_async_engine") as mock_engine:

            mock_settings.return_value = MagicMock(
                QDRANT_URL="http://localhost:6333",
                QDRANT_COLLECTION="test_collection",
                POSTGRES_URL="postgresql+asyncpg://test:test@localhost/test",
            )

            # Create mock search hits — Qdrant returns results sorted by score desc
            mock_hit_1 = MagicMock()
            mock_hit_1.id = "id_2"
            mock_hit_1.score = 0.95
            mock_hit_1.payload = {
                "chunk_id": "chunk_high",
                "content": "Highly relevant content about repo rate.",
            }

            mock_hit_2 = MagicMock()
            mock_hit_2.id = "id_3"
            mock_hit_2.score = 0.85
            mock_hit_2.payload = {
                "chunk_id": "chunk_mid",
                "content": "Moderately relevant content.",
            }

            mock_hit_3 = MagicMock()
            mock_hit_3.id = "id_1"
            mock_hit_3.score = 0.75
            mock_hit_3.payload = {
                "chunk_id": "chunk_low",
                "content": "Lower relevance content.",
            }

            mock_qdrant_instance = MagicMock()
            mock_qdrant_instance.search = AsyncMock(
                return_value=[mock_hit_1, mock_hit_2, mock_hit_3]
            )
            MockQdrant.return_value = mock_qdrant_instance

            from findocs.retrieval.qdrant_client import FinDocsQdrantClient

            client = FinDocsQdrantClient(
                url="http://localhost:6333",
                collection_name="test_collection",
            )

            results = await client.search_dense(
                query_vector=[0.1] * 768,
                top_k=3,
            )

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # Verify scores are in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results should be sorted by score desc, got scores: {scores}"
        )

        # The highest-score result should be first
        assert results[0].chunk_id == "chunk_high", (
            f"Expected chunk_high first, got {results[0].chunk_id}"
        )
