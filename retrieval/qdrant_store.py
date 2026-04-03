from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
import hashlib
import numpy as np
import structlog

from config.config import settings
from chunking.models import ChildChunk

logger = structlog.get_logger()


class QdrantStore:
    """Qdrant vector store supporting dense + sparse vectors."""

    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.create_collection()

    def create_collection(self):
        """Create collection with both dense and sparse vector configs."""
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info("qdrant_collection_created", collection=self.collection)

    def index_chunks(self, chunks: list[ChildChunk], dense_embeddings: np.ndarray,
                     sparse_vectors: list[dict]):
        """Index child chunks with both dense and sparse vectors."""
        points = []
        for i, chunk in enumerate(chunks):
            # Deterministic UUID from chunk.id to avoid collisions across documents
            point_id = hashlib.md5(chunk.id.encode()).hexdigest()
            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_vectors[i]["indices"],
                        values=sparse_vectors[i]["values"],
                    ),
                },
                payload={
                    "chunk_id": chunk.id,
                    "parent_id": chunk.parent_id,
                    "content": chunk.content,
                    "document_id": chunk.metadata.document_id,
                    "citation": chunk.metadata.citation or "",
                    "doc_type": chunk.metadata.doc_type,
                    "court": chunk.metadata.court,
                    "date": chunk.metadata.date,
                    "heading_context": chunk.metadata.heading_context,
                },
            ))

        # Batch upsert (100 at a time)
        batch_size = 100
        for j in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[j:j + batch_size],
            )

        logger.info("chunks_indexed", count=len(points), collection=self.collection)

    def dense_search(self, query_embedding: np.ndarray, top_k: int = 50,
                     doc_type_filter: str = None) -> list[dict]:
        """Dense vector search."""
        filters = None
        if doc_type_filter:
            filters = models.Filter(
                must=[models.FieldCondition(
                    key="doc_type", match=models.MatchValue(value=doc_type_filter)
                )]
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding.tolist(),
            using="dense",
            limit=top_k,
            query_filter=filters,
        )
        return [{"id": r.id, "score": r.score, **r.payload} for r in results.points]

    def sparse_search(self, sparse_vector: dict, top_k: int = 50,
                      doc_type_filter: str = None) -> list[dict]:
        """Sparse vector search."""
        filters = None
        if doc_type_filter:
            filters = models.Filter(
                must=[models.FieldCondition(
                    key="doc_type", match=models.MatchValue(value=doc_type_filter)
                )]
            )

        results = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using="sparse",
            limit=top_k,
            query_filter=filters,
        )
        return [{"id": r.id, "score": r.score, **r.payload} for r in results.points]

    def scroll_all(self) -> list:
        """Return all points (payload only, no vectors) for BM25 corpus rebuild."""
        points = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
        return points

    def update_sparse_vectors(self, updates: list[tuple[str, dict]]):
        """Re-upload sparse vectors for existing points.

        Args:
            updates: list of (point_id_hex, sparse_vector_dict) pairs.
        """
        batch_size = 100
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            self.client.update_vectors(
                collection_name=self.collection,
                points=[
                    models.PointVectors(
                        id=point_id,
                        vector={
                            "sparse": models.SparseVector(
                                indices=sv["indices"],
                                values=sv["values"],
                            )
                        },
                    )
                    for point_id, sv in batch
                ],
            )
