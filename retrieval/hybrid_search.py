from sentence_transformers import CrossEncoder
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
import structlog

from config.config import settings
from retrieval.embedder import EmbeddingService
from retrieval.qdrant_store import QdrantStore
from retrieval.bm25 import BM25Encoder

logger = structlog.get_logger()


class HybridSearchPipeline:
    """Dense -> Sparse -> RRF Fusion -> Reranking -> Parent Hydration"""

    def __init__(self):
        self.embedder = EmbeddingService()
        self.qdrant = QdrantStore()
        self.bm25 = BM25Encoder()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.engine = create_async_engine(settings.postgres_url)

    async def search(self, query: str, doc_type_filter: str = None,
                     top_k: int = None) -> list[dict]:
        """Full hybrid search pipeline."""
        top_k = top_k or settings.final_top_k

        # Step 1: Dense search
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.qdrant.dense_search(
            query_embedding, settings.dense_top_k, doc_type_filter
        )

        # Step 2: Sparse search
        sparse_query = self.bm25.encode_query(query)
        sparse_results = self.qdrant.sparse_search(
            sparse_query, settings.sparse_top_k, doc_type_filter
        )

        # Step 3: RRF Fusion
        fused = self._rrf_fusion(dense_results, sparse_results, k=settings.rrf_k)

        # Step 4: Reranking (top 20 -> top-k)
        reranked = self._rerank(query, fused[:settings.rerank_top_k])

        # Step 5: Parent hydration
        final = await self._hydrate_parents(reranked[:top_k])

        logger.info("hybrid_search_complete",
                     query=query[:80], dense=len(dense_results),
                     sparse=len(sparse_results), fused=len(fused),
                     final=len(final))

        return final

    def _rrf_fusion(self, dense: list[dict], sparse: list[dict], k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion. Combines by rank position, not score."""
        scores = {}
        all_items = {}

        for rank, item in enumerate(dense):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            all_items[chunk_id] = item

        for rank, item in enumerate(sparse):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            all_items[chunk_id] = item

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        return [
            {**all_items[cid], "rrf_score": scores[cid]}
            for cid in sorted_ids
        ]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Cross-encoder reranking."""
        if not candidates:
            return []

        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    async def _hydrate_parents(self, results: list[dict]) -> list[dict]:
        """Fetch parent chunks from Postgres for LLM context."""
        if not results:
            return []

        parent_ids = list(set(r["parent_id"] for r in results))

        async with AsyncSession(self.engine) as session:
            placeholders = ", ".join(f":p{i}" for i in range(len(parent_ids)))
            params = {f"p{i}": pid for i, pid in enumerate(parent_ids)}

            result = await session.execute(
                text(f"SELECT id, content, metadata FROM parent_chunks WHERE id IN ({placeholders})"),
                params,
            )
            parent_map = {row[0]: {"content": row[1], "metadata": row[2]} for row in result.fetchall()}

        # Enrich results with parent content
        for r in results:
            parent = parent_map.get(r["parent_id"], {})
            r["parent_content"] = parent.get("content", r["content"])
            r["parent_metadata"] = parent.get("metadata", {})

        return results
