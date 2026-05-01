from pathlib import Path
from sentence_transformers import CrossEncoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from config.config import settings, get_db_engine
from retrieval.embedder import EmbeddingService
from retrieval.qdrant_store import QdrantStore
from retrieval.bm25 import BM25Encoder

logger = structlog.get_logger()

_BM25_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_encoder.pkl"

# Module-level singleton — CrossEncoder takes ~500 MB and 3-5 s to load.
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


class HybridSearchPipeline:
    """Dense -> Sparse -> RRF Fusion -> Reranking -> Parent Hydration"""

    def __init__(self):
        self.embedder = EmbeddingService()
        self.qdrant = QdrantStore()
        self.bm25 = BM25Encoder()
        if _BM25_PATH.exists():
            self.bm25.load(str(_BM25_PATH))
            logger.info("bm25_loaded", path=str(_BM25_PATH), vocab_size=len(self.bm25.vocab))
        else:
            logger.warning("bm25_encoder_not_found",
                           path=str(_BM25_PATH),
                           hint="run: python scripts/build_bm25_index.py")

    async def search(self, query: str, doc_type_filter: str = None,
                     top_k: int = None) -> list[dict]:
        """Full hybrid search pipeline."""
        top_k = top_k or settings.final_top_k

        query_embedding = self.embedder.embed_query(query)
        dense_results = self.qdrant.dense_search(
            query_embedding, settings.dense_top_k, doc_type_filter
        )

        sparse_query = self.bm25.encode_query(query)
        sparse_results = self.qdrant.sparse_search(
            sparse_query, settings.sparse_top_k, doc_type_filter
        )

        fused = self._rrf_fusion(dense_results, sparse_results, k=settings.rrf_k)

        reranked = self._rerank(query, fused[:settings.rerank_top_k])

        # Keep highest-scoring chunk per document so multiple chunks from the
        # same judgment don't crowd out other results.
        seen_docs: set[str] = set()
        deduped_reranked: list[dict] = []
        for chunk in reranked:
            doc_id = chunk.get("document_id", "")
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                deduped_reranked.append(chunk)

        final = await self._hydrate_parents(deduped_reranked[:top_k])

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

        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        return [
            {**all_items[cid], "rrf_score": scores[cid]}
            for cid in sorted_ids
        ]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Cross-encoder reranking."""
        if not candidates:
            return []

        reranker = _get_reranker()
        pairs = [(query, c["content"]) for c in candidates]
        scores = reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    async def _hydrate_parents(self, results: list[dict]) -> list[dict]:
        """Fetch parent chunks from Postgres for LLM context."""
        if not results:
            return []

        parent_ids = list(set(r["parent_id"] for r in results))

        async with AsyncSession(get_db_engine()) as session:
            placeholders = ", ".join(f":p{i}" for i in range(len(parent_ids)))
            params = {f"p{i}": pid for i, pid in enumerate(parent_ids)}

            result = await session.execute(
                text(f"SELECT id, content, metadata FROM parent_chunks WHERE id IN ({placeholders})"),
                params,
            )
            parent_map = {row[0]: {"content": row[1], "metadata": row[2]} for row in result.fetchall()}

        for r in results:
            parent = parent_map.get(r["parent_id"], {})
            r["parent_content"] = parent.get("content", r["content"])
            r["parent_metadata"] = parent.get("metadata", {})

        return results
