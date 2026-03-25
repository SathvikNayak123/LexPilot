"""Metadata enrichment for document chunks.

After chunking, each :class:`Chunk` object contains raw text but is missing
document-level metadata (source path, document type, publication date, and
heading context).  :class:`MetadataEnricher` fills those fields by
cross-referencing the originating :class:`ParsedDocument`.
"""

from __future__ import annotations

import structlog

from findocs.ingestion.models import ParsedDocument
from findocs.processing.chunker import Chunk

logger = structlog.get_logger(__name__)


class MetadataEnricher:
    """Enrich a list of chunks with source-document metadata.

    This class is intentionally stateless; all information needed for
    enrichment comes from the *ParsedDocument* passed to :meth:`enrich`.
    """

    def enrich(self, chunks: list[Chunk], doc: ParsedDocument) -> list[Chunk]:
        """Attach document-level metadata to every chunk.

        Fields populated:
            * ``doc_source`` -- string representation of the source file path.
            * ``doc_type`` -- document type identifier (e.g. ``rbi_circular``).
            * ``doc_date`` -- ISO-formatted publication date, if available.
            * ``headings_context`` -- concatenated heading texts that precede
              the chunk's page, providing hierarchical context for the LLM.

        Args:
            chunks: List of chunks produced by the
                :class:`~findocs.processing.chunker.SemanticChunker`.
            doc: The parsed document from which the chunks were derived.

        Returns:
            The **same** list of chunks, mutated in place with metadata
            fields set.  Returning the list is a convenience for chaining.
        """
        log = logger.bind(source=str(doc.source_path), doc_type=doc.doc_type)
        log.info("metadata_enrich_start", num_chunks=len(chunks))

        doc_source = str(doc.source_path)
        doc_type = doc.doc_type
        doc_date = doc.date.isoformat() if doc.date else None

        # Pre-build a page -> headings mapping from the document blocks
        page_headings = self._build_page_headings(doc)

        for chunk in chunks:
            chunk.doc_source = doc_source
            chunk.doc_type = doc_type
            chunk.doc_date = doc_date

            # Gather headings for the chunk's page and all preceding pages so
            # the LLM receives full hierarchical context.
            chunk.headings_context = self._resolve_headings(chunk.page_num, page_headings)

        log.info("metadata_enrich_done", num_chunks=len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_page_headings(doc: ParsedDocument) -> dict[int, list[str]]:
        """Extract heading texts from parsed blocks, grouped by page number.

        A block is considered a heading when its ``metadata`` dictionary
        contains ``"is_heading": True``.

        Args:
            doc: The source parsed document.

        Returns:
            Dictionary mapping page numbers to an ordered list of heading
            strings found on that page.
        """
        headings: dict[int, list[str]] = {}
        for block in doc.blocks:
            is_heading = block.metadata.get("is_heading", False)
            if is_heading:
                headings.setdefault(block.page_num, []).append(block.content.strip())
        return headings

    @staticmethod
    def _resolve_headings(
        page_num: int | None,
        page_headings: dict[int, list[str]],
    ) -> str:
        """Build a heading-context string for a given page number.

        All headings from page 1 up to and including *page_num* are
        concatenated with `` > `` separators, producing a breadcrumb-like
        trail (e.g. ``"Annual Report > Financial Statements > Balance Sheet"``).

        Args:
            page_num: The page number of the chunk.  If *None*, an empty
                string is returned.
            page_headings: Mapping from page number to headings on that page.

        Returns:
            Concatenated heading string, or ``""`` if no headings are found.
        """
        if page_num is None:
            return ""

        collected: list[str] = []
        for pn in sorted(page_headings.keys()):
            if pn > page_num:
                break
            collected.extend(page_headings[pn])

        return " > ".join(collected)
