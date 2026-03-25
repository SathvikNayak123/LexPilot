"""Qdrant vector database client for FinDocs hybrid retrieval.

Manages a Qdrant collection with both dense (all-mpnet-base-v2) and sparse
(BM25 IDF-weighted) vectors.  Only *child* chunks are stored in Qdrant;
parent chunks live in Postgres and are fetched on demand via ``parent_id``.
"""

from __future__ import annotations

# ARCHITECTURE DECISION: Why only child chunks in Qdrant, parents in Postgres
# Child chunks (128 tokens) provide fine-grained retrieval — they match specific
# facts or data points. But 128 tokens is too little context for LLM generation.
# Parent chunks (512 tokens) stored in Postgres provide the broader context window.
# This separation gives us retrieval precision (search on children) while maintaining
# generation quality (prompt with parents). The parent_id foreign key links them.

import uuid
from typing import Any

import structlog
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from config.config import get_settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DENSE_DIM: int = 768
DENSE_DISTANCE: Distance = Distance.COSINE
SPARSE_VECTOR_NAME: str = "bm25"
UPSERT_BATCH_SIZE: int = 100


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """Represents a text chunk stored in the system."""

    chunk_id: str = Field(description="Unique identifier for the chunk.")
    content: str = Field(description="Text content of the chunk.")
    parent_id: str | None = Field(
        default=None,
        description="Foreign key to the parent chunk in Postgres (child chunks only).",
    )
    doc_type: str | None = Field(default=None, description="Document type label.")
    doc_date: str | None = Field(default=None, description="Document date as ISO-8601 string.")
    page_num: int | None = Field(default=None, description="Source page number.")
    chunk_type: str = Field(default="child", description="'parent' or 'child'.")


class SearchResult(BaseModel):
    """A single search result returned from Qdrant."""

    chunk_id: str = Field(description="Identifier of the matched chunk.")
    content: str = Field(description="Text content of the matched chunk.")
    score: float = Field(description="Similarity / relevance score.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata payload.")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FinDocsQdrantClient:
    """High-level async wrapper around Qdrant for FinDocs retrieval.

    Parameters
    ----------
    url:
        Qdrant gRPC / HTTP endpoint (e.g. ``http://localhost:6333``).
    collection_name:
        Name of the Qdrant collection to manage.
    """

    def __init__(self, url: str | None = None, collection_name: str | None = None) -> None:
        settings = get_settings()
        self._url: str = url or settings.QDRANT_URL
        self._collection: str = collection_name or settings.QDRANT_COLLECTION
        self._client: AsyncQdrantClient = AsyncQdrantClient(url=self._url)

        # Postgres engine for parent-chunk lookups
        self._pg_engine = create_async_engine(settings.POSTGRES_URL, pool_size=5, max_overflow=10)

        logger.info(
            "qdrant_client.init",
            url=self._url,
            collection=self._collection,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def create_collection(self) -> None:
        """Create the Qdrant collection with dense + sparse vector configuration.

        If the collection already exists, this method is a no-op and logs a
        warning instead of raising.
        """

        collections = await self._client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if self._collection in existing_names:
            logger.warning("qdrant_client.collection_exists", collection=self._collection)
            return

        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=DENSE_DISTANCE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )

        logger.info(
            "qdrant_client.collection_created",
            collection=self._collection,
            dense_dim=DENSE_DIM,
            distance=DENSE_DISTANCE.value,
        )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
    ) -> None:
        """Upsert child chunks (with dense + sparse vectors) in batches.

        Only chunks whose ``chunk_type`` is ``'child'`` are stored in Qdrant.
        Parent chunks are expected to live in Postgres.

        Parameters
        ----------
        chunks:
            The child ``Chunk`` objects to upsert.
        dense_vectors:
            Dense embedding vectors aligned 1-to-1 with *chunks*.
        sparse_vectors:
            BM25 sparse vectors aligned 1-to-1 with *chunks*.

        Raises
        ------
        ValueError
            If the input lists have mismatched lengths.
        """

        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError(
                f"Length mismatch: chunks={len(chunks)}, "
                f"dense={len(dense_vectors)}, sparse={len(sparse_vectors)}"
            )

        # Filter to child chunks only
        child_indices = [i for i, c in enumerate(chunks) if c.chunk_type == "child"]
        if not child_indices:
            logger.warning("qdrant_client.upsert_chunks.no_children")
            return

        points: list[PointStruct] = []
        for idx in child_indices:
            chunk = chunks[idx]
            point = PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector={
                    "dense": dense_vectors[idx],
                    SPARSE_VECTOR_NAME: sparse_vectors[idx],
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "parent_id": chunk.parent_id,
                    "doc_type": chunk.doc_type,
                    "doc_date": chunk.doc_date,
                    "page_num": chunk.page_num,
                    "chunk_type": chunk.chunk_type,
                },
            )
            points.append(point)

        total = len(points)
        for start in range(0, total, UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            await self._client.upsert(collection_name=self._collection, points=batch)
            logger.info(
                "qdrant_client.upsert_batch",
                batch_start=start,
                batch_size=len(batch),
                total=total,
            )

        logger.info("qdrant_client.upsert_complete", total_upserted=total)

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    async def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the collection using the dense (cosine) vector.

        Parameters
        ----------
        query_vector:
            768-dim dense embedding for the query.
        top_k:
            Maximum number of results to return.
        filters:
            Optional Qdrant filter expressed as ``{field: value}`` pairs
            (each translated to an exact-match condition).

        Returns
        -------
        list[SearchResult]
            Ranked search results with scores.
        """

        qdrant_filter: Filter | None = None
        if filters:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=NamedVector(name="dense", vector=query_vector),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    content=payload.get("content", ""),
                    score=hit.score,
                    payload=payload,
                )
            )

        logger.info("qdrant_client.search_dense", top_k=top_k, returned=len(results))
        return results

    # ------------------------------------------------------------------
    # Sparse search
    # ------------------------------------------------------------------

    async def search_sparse(
        self,
        sparse_vector: SparseVector,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Search the collection using the BM25 sparse vector.

        Parameters
        ----------
        sparse_vector:
            A ``SparseVector`` (indices + values) representing the BM25 query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked search results with BM25 scores.
        """

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=NamedSparseVector(
                name=SPARSE_VECTOR_NAME,
                vector=sparse_vector,
            ),
            limit=top_k,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    content=payload.get("content", ""),
                    score=hit.score,
                    payload=payload,
                )
            )

        logger.info("qdrant_client.search_sparse", top_k=top_k, returned=len(results))
        return results

    # ------------------------------------------------------------------
    # Parent-chunk fetch (Postgres)
    # ------------------------------------------------------------------

    async def get_parent_chunk(self, parent_id: str) -> Chunk | None:
        """Fetch a parent chunk from Postgres by its ID.

        Parameters
        ----------
        parent_id:
            Primary key of the parent chunk row.

        Returns
        -------
        Chunk | None
            The parent ``Chunk`` if found, otherwise ``None``.
        """

        query = text(
            "SELECT chunk_id, content, parent_id, doc_type, doc_date, page_num, chunk_type "
            "FROM chunks WHERE chunk_id = :cid AND chunk_type = 'parent' LIMIT 1"
        )

        async with AsyncSession(self._pg_engine) as session:
            result = await session.execute(query, {"cid": parent_id})
            row = result.mappings().first()

        if row is None:
            logger.warning("qdrant_client.parent_not_found", parent_id=parent_id)
            return None

        chunk = Chunk(
            chunk_id=row["chunk_id"],
            content=row["content"],
            parent_id=row["parent_id"],
            doc_type=row["doc_type"],
            doc_date=str(row["doc_date"]) if row["doc_date"] else None,
            page_num=row["page_num"],
            chunk_type=row["chunk_type"],
        )

        logger.debug("qdrant_client.parent_fetched", parent_id=parent_id)
        return chunk
