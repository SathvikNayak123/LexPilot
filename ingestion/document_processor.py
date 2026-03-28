import json
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

from config.config import settings
from ingestion.pdf_parser import PDFParser
from ingestion.models import ParsedDocument

logger = structlog.get_logger()


class DocumentProcessor:
    """Orchestrates document ingestion: parse PDF -> store metadata in Postgres."""

    def __init__(self):
        self.parser = PDFParser()
        self.engine = create_async_engine(settings.postgres_url)

    async def ingest(self, pdf_path: str, doc_type: str = "judgment",
                     source: str = None, court: str = None) -> ParsedDocument:
        """Parse a PDF and store its metadata."""
        doc = self.parser.parse(pdf_path, doc_type, source, court)

        async with AsyncSession(self.engine) as session:
            await session.execute(
                text("""
                    INSERT INTO documents (id, title, doc_type, source, court, date, metadata, parse_duration_ms)
                    VALUES (:id, :title, :dt, :src, :court, :date, :meta, :dur)
                    ON CONFLICT (id) DO UPDATE SET
                        title = :title, metadata = :meta, parse_duration_ms = :dur
                """),
                {
                    "id": doc.document_id, "title": doc.title,
                    "dt": doc.doc_type, "src": doc.source,
                    "court": doc.court, "date": str(doc.date) if doc.date else None,
                    "meta": json.dumps(doc.metadata),
                    "dur": doc.parse_duration_ms,
                },
            )
            await session.commit()

        logger.info("document_ingested", doc_id=doc.document_id,
                     title=doc.title, blocks=len(doc.blocks))
        return doc

    async def ingest_directory(self, directory: str, doc_type: str = "judgment",
                                source: str = None, court: str = None) -> list[ParsedDocument]:
        """Ingest all PDFs in a directory."""
        docs = []
        for pdf_path in sorted(Path(directory).glob("*.pdf")):
            try:
                doc = await self.ingest(str(pdf_path), doc_type, source, court)
                docs.append(doc)
            except Exception as e:
                logger.error("ingestion_failed", path=str(pdf_path), error=str(e))
        return docs
