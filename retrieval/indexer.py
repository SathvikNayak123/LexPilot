from ingestion.document_processor import DocumentProcessor
from chunking.semantic_chunker import SemanticChunker
from retrieval.embedder import EmbeddingService
from retrieval.qdrant_store import QdrantStore
from retrieval.bm25 import BM25Encoder
from retrieval.parent_store import ParentChunkStore
from knowledge_graph.graph_builder import GraphBuilder
import structlog

logger = structlog.get_logger()


class IndexingPipeline:
    """Full document -> index pipeline."""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.chunker = SemanticChunker()
        self.embedder = EmbeddingService()
        self.qdrant = QdrantStore()
        self.bm25 = BM25Encoder()
        self.parent_store = ParentChunkStore()
        self.graph_builder = GraphBuilder()

    async def index_document(self, pdf_path: str, doc_type: str = "judgment", **kwargs):
        """Ingest, chunk, embed, and index a single document."""
        # Parse
        doc = await self.processor.ingest(pdf_path, doc_type, **kwargs)

        # Chunk
        parents, children = self.chunker.chunk_document(doc)

        # Store parents in Postgres
        await self.parent_store.store(parents)

        # Embed children (dense) — embarrassingly parallel, no lock needed
        child_texts = [c.content for c in children]
        dense_embeddings = self.embedder.embed(child_texts)

        # Build sparse vectors
        self.bm25.fit(child_texts)
        sparse_vectors = [self.bm25.encode_document(t) for t in child_texts]

        # Index in Qdrant
        self.qdrant.index_chunks(children, dense_embeddings, sparse_vectors)

        # Build knowledge graph (judgments only)
        citations_found = 0
        if doc_type == "judgment":
            try:
                citations_found = await self.graph_builder.build_from_document(doc, kwargs)
            except Exception as e:
                logger.warning("graph_build_skipped", doc_id=doc.document_id, error=str(e))

        return {"parents": len(parents), "children": len(children), "citations": citations_found}
