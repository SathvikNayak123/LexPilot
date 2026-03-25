"""Top-level PDF parser that routes each page to the correct sub-parser.

For every page the parser detects which content types are present (text,
table, image/chart) and dispatches to ``TextExtractor``,
``TableExtractor``, and ``ChartExtractor`` accordingly.  All results are
combined into a single ``ParsedDocument``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import fitz
import structlog

from findocs.ingestion.models import (
    ChartBlock,
    ParsedBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)
from findocs.ingestion.parsers.chart_extractor import ChartExtractor
from findocs.ingestion.parsers.table_extractor import TableExtractor
from findocs.ingestion.parsers.text_extractor import TextExtractor

logger = structlog.get_logger(__name__)

# Minimum characters returned by ``page.get_text()`` to classify as text.
_TEXT_CHAR_THRESHOLD: int = 100

# Minimum horizontal drawing lines to classify as containing a table.
_TABLE_LINE_THRESHOLD: int = 5


class PDFParser:
    """Route PDF pages to sub-parsers based on content-type detection.

    Instantiate once and call :meth:`parse` for each PDF.  The class owns
    the three sub-parsers and manages the per-PDF cost/rate counters of
    the ``ChartExtractor``.

    Args:
        openai_api_key: Optional API key forwarded to ``ChartExtractor``.
    """

    def __init__(self, openai_api_key: str | None = None) -> None:
        self._text_extractor = TextExtractor()
        self._table_extractor = TableExtractor()
        self._chart_extractor = ChartExtractor(openai_api_key=openai_api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(
        self,
        pdf_path: Path,
        doc_metadata: dict,
    ) -> ParsedDocument:
        """Parse a full PDF and return a ``ParsedDocument``.

        Each page is analysed for content types using simple heuristics,
        then the appropriate sub-parsers are called.  Text extraction is
        synchronous; chart extraction is ``async`` (OpenAI API calls).

        Args:
            pdf_path: Filesystem path to the PDF file.
            doc_metadata: Dict containing at least ``doc_type`` and
                ``title``.  Optionally ``date`` (ISO string or
                ``datetime``).

        Returns:
            A ``ParsedDocument`` aggregating all extracted blocks.
        """
        start = time.perf_counter()

        # Reset chart-extractor counters for the new PDF
        self._chart_extractor.reset_counters()

        all_blocks: list[ParsedBlock] = []

        doc = fitz.open(str(pdf_path))
        total_pages = doc.page_count

        logger.info(
            "pdf_parser.start",
            pdf_path=str(pdf_path),
            total_pages=total_pages,
        )

        try:
            for page_idx in range(total_pages):
                page: fitz.Page = doc[page_idx]
                page_num = page_idx + 1

                content_types = self.detect_page_content_types(page)

                logger.debug(
                    "pdf_parser.page_content_types",
                    page_num=page_num,
                    content_types=sorted(content_types),
                )

                # ---- Text extraction ---- #
                if "text" in content_types:
                    text_blocks: list[TextBlock] = self._text_extractor.extract_blocks(page)
                    paragraphs: list[str] = self._text_extractor.reconstruct_paragraphs(text_blocks)
                    for para in paragraphs:
                        if para.strip():
                            all_blocks.append(
                                ParsedBlock(
                                    content=para,
                                    block_type="text",
                                    page_num=page_num,
                                    metadata={
                                        "source": "text_extractor",
                                    },
                                )
                            )

                # ---- Table extraction ---- #
                if "table" in content_types:
                    table_blocks: list[TableBlock] = self._table_extractor.extract_tables(
                        pdf_path, page_num
                    )
                    for tb in table_blocks:
                        metadata: dict[str, object] = {
                            "source": "table_extractor",
                            "caption": tb.caption,
                            "row_count": tb.row_count,
                            "col_count": tb.col_count,
                        }
                        if tb.numerical_summary is not None:
                            metadata["numerical_summary"] = tb.numerical_summary

                        all_blocks.append(
                            ParsedBlock(
                                content=tb.markdown_content,
                                block_type="table",
                                page_num=page_num,
                                metadata=metadata,
                            )
                        )

                # ---- Chart/Image extraction ---- #
                if "image" in content_types:
                    chart_blocks: list[ChartBlock] = await self._chart_extractor.batch_extract(page)
                    for cb in chart_blocks:
                        all_blocks.append(
                            ParsedBlock(
                                content=cb.description,
                                block_type="chart",
                                page_num=page_num,
                                metadata={
                                    "source": "chart_extractor",
                                    "chart_type": cb.chart_type,
                                    "extraction_cost_usd": cb.extraction_cost_usd,
                                },
                            )
                        )
        finally:
            doc.close()

        duration = round(time.perf_counter() - start, 3)

        parsed_doc = ParsedDocument(
            source_path=pdf_path,
            doc_type=doc_metadata.get("doc_type", "rbi_circular"),
            title=doc_metadata.get("title", pdf_path.stem),
            date=doc_metadata.get("date"),
            blocks=all_blocks,
            total_pages=total_pages,
            parsing_duration_seconds=duration,
        )

        logger.info(
            "pdf_parser.complete",
            pdf_path=str(pdf_path),
            total_pages=total_pages,
            total_blocks=len(all_blocks),
            text_blocks=sum(1 for b in all_blocks if b.block_type == "text"),
            table_blocks=sum(1 for b in all_blocks if b.block_type == "table"),
            chart_blocks=sum(1 for b in all_blocks if b.block_type == "chart"),
            duration_s=duration,
        )

        return parsed_doc

    # ------------------------------------------------------------------
    # Content-type detection
    # ------------------------------------------------------------------

    def detect_page_content_types(
        self,
        page: fitz.Page,
    ) -> set[Literal["text", "table", "image"]]:
        """Determine what kinds of content a page contains.

        Heuristics
        ----------
        * **text**: ``page.get_text()`` yields more than 100 characters.
        * **table**: The page's drawing commands contain more than 5
          roughly-horizontal lines (``abs(y1 - y0) < 2``).
        * **image**: ``page.get_images()`` returns at least one entry.

        Args:
            page: A PyMuPDF ``fitz.Page`` object.

        Returns:
            A set of content-type literals (may be empty if the page is
            blank).
        """
        content_types: set[Literal["text", "table", "image"]] = set()

        # --- Text detection --- #
        text = page.get_text()
        if len(text) > _TEXT_CHAR_THRESHOLD:
            content_types.add("text")

        # --- Table detection (line density heuristic) --- #
        try:
            drawings = page.get_drawings()
            horizontal_line_count = 0
            for drawing in drawings:
                for item in drawing.get("items", []):
                    # Each item is a tuple like ("l", p1, p2) for a line
                    if len(item) >= 3 and item[0] == "l":
                        p1 = item[1]
                        p2 = item[2]
                        # Horizontal if y-coordinates are close
                        if abs(p1.y - p2.y) < 2:
                            horizontal_line_count += 1

            if horizontal_line_count > _TABLE_LINE_THRESHOLD:
                content_types.add("table")
        except Exception:
            logger.debug(
                "pdf_parser.drawing_detection_error",
                page_num=page.number + 1,
            )

        # --- Image detection --- #
        images = page.get_images()
        if len(images) > 0:
            content_types.add("image")

        return content_types
