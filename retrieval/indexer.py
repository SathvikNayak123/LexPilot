import asyncio
from typing import List

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

from config.config import settings
from ingestion.document_processor import DocumentProcessor
from chunking.semantic_chunker import SemanticChunker
from retrieval.embedder import EmbeddingService
from retrieval.qdrant_store import QdrantStore
from retrieval.bm25 import BM25Encoder
from retrieval.parent_store import ParentChunkStore
import structlog

logger = structlog.get_logger()


def _bm25_fit_encode(bm25: BM25Encoder, texts: List[str]) -> List[dict]:
    """Fit BM25 on texts and encode each one. Pure function — safe to run in a thread."""
    bm25.fit(texts)
    return [bm25.encode_document(t) for t in texts]


class IndexingPipeline:
    """Full document -> index pipeline."""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.chunker = SemanticChunker()
        self.embedder = EmbeddingService()
        self.qdrant = QdrantStore()
        self.parent_store = ParentChunkStore()
        self.engine = create_async_engine(settings.postgres_url)

    async def index_document(self, pdf_path: str, doc_type: str = "judgment", **kwargs):
        """Ingest, chunk, embed, and index a single document."""
        # Parse (sync fitz+pdfplumber already offloaded to thread in DocumentProcessor)
        doc = await self.processor.ingest(
            pdf_path, doc_type,
            source=kwargs.get("source"),
            court=kwargs.get("court"),
            citation=kwargs.get("citation"),
        )
        # Attach extra metadata for citation_index and graph builder
        if kwargs.get("holding_summary"):
            doc.metadata["holding_summary"] = kwargs["holding_summary"]
        if kwargs.get("is_overruled"):
            doc.metadata["is_overruled"] = kwargs["is_overruled"]
        if kwargs.get("overruled_by"):
            doc.metadata["overruled_by"] = kwargs["overruled_by"]

        # Chunk — spaCy + SentenceTransformer are sync/CPU-bound; run in thread pool.
        parents, children = await asyncio.to_thread(self.chunker.chunk_document, doc)

        # Store parents in Postgres
        await self.parent_store.store(parents)

        child_texts = [c.content for c in children]

        # Embed (dense) and build sparse vectors concurrently.
        # BM25Encoder is instantiated locally — shared self.bm25 would be a race condition
        # when multiple documents are indexed in parallel (fit() mutates vocabulary state).
        bm25 = BM25Encoder()

        dense_embeddings, sparse_vectors = await asyncio.gather(
            asyncio.to_thread(self.embedder.embed, child_texts),
            asyncio.to_thread(_bm25_fit_encode, bm25, child_texts),
        )

        # Index in Qdrant (sync client — run in thread pool)
        await asyncio.to_thread(self.qdrant.index_chunks, children, dense_embeddings, sparse_vectors)

        # Populate citation_index for the citation verifier.
        # Graph nodes + edges are built separately via scripts/build_semantic_graph.py
        # after all documents are ingested — this avoids dropped edges from ingestion order.
        if doc.citation and doc_type == "judgment":
            await self._index_citation(doc)

        return {"parents": len(parents), "children": len(children)}

    async def _index_citation(self, doc):
        """Insert/update citation_index row for citation verification."""
        async with AsyncSession(self.engine) as session:
            await session.execute(
                text("""
                    INSERT INTO citation_index (citation_string, case_name, court, date,
                                                holding_summary, is_overruled, overruled_by)
                    VALUES (:cit, :name, :court, :date, :holding, :overruled, :overruled_by)
                    ON CONFLICT (citation_string) DO UPDATE SET
                        case_name = :name, court = :court, holding_summary = :holding,
                        is_overruled = :overruled, overruled_by = :overruled_by
                """),
                {
                    "cit": doc.citation, "name": doc.title,
                    "court": doc.court or "Unknown",
                    "date": str(doc.date) if doc.date else None,
                    "holding": doc.metadata.get("holding_summary"),
                    "overruled": doc.metadata.get("is_overruled", False),
                    "overruled_by": doc.metadata.get("overruled_by"),
                },
            )
            await session.commit()
