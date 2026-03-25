"""Hybrid dense + sparse search with RRF fusion and cross-encoder reranking.

Orchestrates the full retrieval pipeline: embed the query, run dense and BM25
searches in parallel, fuse via Reciprocal Rank Fusion, deduplicate, rerank
with a cross-encoder, and finally hydrate parent chunks from Postgres.
"""

from __future__ import annotations

# ARCHITECTURE DECISION: Why RRF over linear interpolation for fusion
# Linear interpolation (α*dense + (1-α)*sparse) requires careful tuning of α and
# assumes both score distributions are comparable — they rarely are. Dense cosine
# scores cluster in [0.3, 0.8] while BM25 scores can range [0, 30+]. RRF (Reciprocal
# Rank Fusion) operates on ranks, not scores, making it robust to scale differences.
# Formula: score(d) = Σ 1/(k + rank(d, list)) with k=60. No hyperparameter tuning needed.

import asyncio
from collections import defaultdict
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, Field
from qdrant_client.models import SparseVector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.config import get_settings
from retrieval.qdrant_client import Chunk, FinDocsQdrantClient, SearchResult
from retrieval.reranker import Reranker

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RetrievedContext(BaseModel):
    """A fully hydrated retrieval result ready for LLM prompt assembly."""

    child_chunk: Chunk = Field(description="The fine-grained child chunk that matched the query.")
    parent_chunk: Chunk | None = Field(
        default=None,
        description="The broader parent chunk from Postgres (512 tokens).",
    )
    dense_score: float = Field(default=0.0, description="Dense cosine similarity score.")
    sparse_score: float = Field(default=0.0, description="BM25 sparse retrieval score.")
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion score.")
    reranker_score: float = Field(default=0.0, description="Cross-encoder reranker score.")
    final_context: str = Field(
        default="",
        description="Text sent to the LLM — parent content if available, else child content.",
    )


# ---------------------------------------------------------------------------
# BM25 sparse-vector helper
# ---------------------------------------------------------------------------


class _BM25SparseEncoder:
    """Builds IDF-weighted sparse vectors compatible with Qdrant.

    The encoder maintains a vocabulary fitted on a corpus. At query time it
    converts a tokenised query into a ``SparseVector`` of ``(index, idf_weight)``
    pairs.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._fitted: bool = False

    def fit(self, corpus_tokens: list[list[str]]) -> None:
        """Fit BM25 IDF values from a tokenised corpus.

        Parameters
        ----------
        corpus_tokens:
            List of tokenised documents (each document is a list of tokens).
        """

        bm25 = BM25Okapi(corpus_tokens)
        all_tokens: set[str] = set()
        for doc in corpus_tokens:
            all_tokens.update(doc)

        self._vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        n_docs = len(corpus_tokens)
        for token in all_tokens:
            df = bm25.doc_freqs.get(token, 0)
            self._idf[token] = float(np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0))

        self._fitted = True
        logger.info("bm25_encoder.fitted", vocab_size=len(self._vocab))

    def encode_query(self, query_tokens: list[str]) -> SparseVector:
        """Encode a tokenised query into a Qdrant ``SparseVector``.

        Parameters
        ----------
        query_tokens:
            Whitespace-split query tokens (lowercased).

        Returns
        -------
        SparseVector
            Sparse vector with IDF-weighted values for known tokens.
        """

        indices: list[int] = []
        values: list[float] = []
        seen: set[str] = set()

        for token in query_tokens:
            if token in seen:
                continue
            seen.add(token)
            if token in self._vocab:
                indices.append(self._vocab[token])
                values.append(self._idf.get(token, 0.0))

        if not indices:
            logger.warning("bm25_encoder.empty_vector", tokens=query_tokens)
            return SparseVector(indices=[0], values=[0.0])

        return SparseVector(indices=indices, values=values)


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------


class HybridSearcher:
    """End-to-end hybrid retrieval: dense + sparse search, RRF fusion, reranking.

    Parameters
    ----------
    qdrant_client:
        Initialised ``FinDocsQdrantClient`` instance.
    reranker:
        Initialised ``Reranker`` instance.
    embedding_model_name:
        HuggingFace model id for the dense embedder.  Defaults to
        ``settings.EMBEDDING_MODEL``.
    corpus_tokens:
        Pre-tokenised corpus used to fit BM25 IDF weights.  If ``None``,
        the BM25 encoder starts unfitted and ``search`` will produce
        zero-weight sparse vectors until ``fit_bm25`` is called.
    """

    def __init__(
        self,
        qdrant_client: FinDocsQdrantClient | None = None,
        reranker: Reranker | None = None,
        embedding_model_name: str | None = None,
        corpus_tokens: list[list[str]] | None = None,
    ) -> None:
        settings = get_settings()

        self._qdrant = qdrant_client or FinDocsQdrantClient()
        self._reranker = reranker or Reranker()

        model_name = embedding_model_name or settings.EMBEDDING_MODEL
        logger.info("hybrid_searcher.loading_embedding_model", model=model_name)
        self._embedder = SentenceTransformer(model_name)
        logger.info("hybrid_searcher.embedding_model_loaded", model=model_name)

        self._bm25_encoder = _BM25SparseEncoder()
        if corpus_tokens is not None:
            self._bm25_encoder.fit(corpus_tokens)

        self._settings = settings

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def fit_bm25(self, corpus_tokens: list[list[str]]) -> None:
        """Fit the BM25 sparse encoder on a tokenised corpus.

        Parameters
        ----------
        corpus_tokens:
            List of tokenised documents.
        """

        self._bm25_encoder.fit(corpus_tokens)

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type_filter: str | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> list[RetrievedContext]:
        """Execute the full hybrid retrieval pipeline.

        Steps
        -----
        1. Embed query → dense vector.
        2. Tokenise query → BM25 sparse vector.
        3. Run dense + sparse searches in parallel (``asyncio.gather``).
        4. RRF fusion with *k=60*.
        5. Deduplicate by ``chunk_id``.
        6. Rerank top 20 with cross-encoder.
        7. Fetch parent chunks for top *top_k* results.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Number of final results to return (after reranking).
        doc_type_filter:
            If provided, restrict dense search to this document type.
        date_range:
            ``(start_iso, end_iso)`` date range filter (applied via payload).

        Returns
        -------
        list[RetrievedContext]
            Fully hydrated retrieval contexts ready for LLM prompting.
        """

        logger.info("hybrid_search.start", query=query, top_k=top_k)

        # 1. Dense embedding
        dense_vector: list[float] = self._embedder.encode(query).tolist()  # type: ignore[union-attr]

        # 2. Sparse BM25 vector
        query_tokens = query.lower().split()
        sparse_vector = self._bm25_encoder.encode_query(query_tokens)

        # Build optional filters
        filters: dict[str, Any] | None = None
        if doc_type_filter or date_range:
            filters = {}
            if doc_type_filter:
                filters["doc_type"] = doc_type_filter
            # date_range filtering is handled post-hoc below if the store
            # doesn't support range filters natively.

        # 3. Parallel dense + sparse search
        dense_task = self._qdrant.search_dense(
            query_vector=dense_vector,
            top_k=self._settings.DENSE_TOP_K,
            filters=filters,
        )
        sparse_task = self._qdrant.search_sparse(
            sparse_vector=sparse_vector,
            top_k=self._settings.SPARSE_TOP_K,
        )

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # Optional date-range post-filter
        if date_range:
            start, end = date_range
            dense_results = _filter_by_date(dense_results, start, end)
            sparse_results = _filter_by_date(sparse_results, start, end)

        logger.info(
            "hybrid_search.raw_results",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        # 4. RRF fusion
        fused = self.reciprocal_rank_fusion(dense_results, sparse_results, k=60)

        # 5. Deduplicate by chunk_id
        seen_ids: set[str] = set()
        deduped: list[tuple[str, float]] = []
        for chunk_id, score in fused:
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                deduped.append((chunk_id, score))

        # Build lookup maps for scores
        dense_score_map: dict[str, float] = {r.chunk_id: r.score for r in dense_results}
        sparse_score_map: dict[str, float] = {r.chunk_id: r.score for r in sparse_results}
        rrf_score_map: dict[str, float] = dict(deduped)

        # Collect SearchResult objects for reranking (top 20)
        all_results_map: dict[str, SearchResult] = {}
        for r in dense_results:
            all_results_map[r.chunk_id] = r
        for r in sparse_results:
            if r.chunk_id not in all_results_map:
                all_results_map[r.chunk_id] = r

        candidates: list[SearchResult] = []
        for chunk_id, _ in deduped[:20]:
            if chunk_id in all_results_map:
                candidates.append(all_results_map[chunk_id])

        # 6. Rerank top 20
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)

        # 7. Fetch parent chunks for top results
        contexts: list[RetrievedContext] = []
        parent_tasks = []
        for result in reranked:
            parent_id = result.payload.get("parent_id")
            if parent_id:
                parent_tasks.append(self._qdrant.get_parent_chunk(parent_id))
            else:
                parent_tasks.append(_null_coroutine())

        parent_chunks: list[Chunk | None] = await asyncio.gather(*parent_tasks)

        for result, parent in zip(reranked, parent_chunks, strict=True):
            child = Chunk(
                chunk_id=result.chunk_id,
                content=result.content,
                parent_id=result.payload.get("parent_id"),
                doc_type=result.payload.get("doc_type"),
                doc_date=result.payload.get("doc_date"),
                page_num=result.payload.get("page_num"),
                chunk_type="child",
            )

            final_context = parent.content if parent else child.content

            ctx = RetrievedContext(
                child_chunk=child,
                parent_chunk=parent,
                dense_score=dense_score_map.get(result.chunk_id, 0.0),
                sparse_score=sparse_score_map.get(result.chunk_id, 0.0),
                rrf_score=rrf_score_map.get(result.chunk_id, 0.0),
                reranker_score=result.score,
                final_context=final_context,
            )
            contexts.append(ctx)

        logger.info("hybrid_search.complete", returned=len(contexts))
        return contexts

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Fuse two ranked lists using Reciprocal Rank Fusion (RRF).

        For each document *d*, the RRF score is:

            ``score(d) = Σ  1 / (k + rank(d, list_i))``

        where *rank* starts at 1 for the top result.

        Parameters
        ----------
        dense_results:
            Ranked results from dense search.
        sparse_results:
            Ranked results from sparse search.
        k:
            RRF constant (default 60, per the original paper).

        Returns
        -------
        list[tuple[str, float]]
            ``(chunk_id, rrf_score)`` pairs sorted by descending RRF score.
        """

        scores: dict[str, float] = defaultdict(float)

        for rank, result in enumerate(dense_results, start=1):
            scores[result.chunk_id] += 1.0 / (k + rank)

        for rank, result in enumerate(sparse_results, start=1):
            scores[result.chunk_id] += 1.0 / (k + rank)

        fused = sorted(scores.items(), key=lambda t: t[1], reverse=True)

        logger.info(
            "rrf.fused",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            fused_count=len(fused),
        )

        return fused


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_by_date(results: list[SearchResult], start: str, end: str) -> list[SearchResult]:
    """Filter results whose ``doc_date`` falls within [start, end].

    Parameters
    ----------
    results:
        Search results to filter.
    start:
        ISO-8601 start date (inclusive).
    end:
        ISO-8601 end date (inclusive).

    Returns
    -------
    list[SearchResult]
        Filtered subset.
    """

    filtered: list[SearchResult] = []
    for r in results:
        doc_date = r.payload.get("doc_date")
        if doc_date is None:
            continue
        if start <= str(doc_date) <= end:
            filtered.append(r)
    return filtered


async def _null_coroutine() -> None:
    """Return ``None`` as an awaitable — used when no parent fetch is needed."""
    return None
