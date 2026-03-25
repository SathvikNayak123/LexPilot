"""PyMuPDF-based text extraction with structure preservation.

Extracts text blocks from PDF pages, detects headings by font size,
and reconstructs logical paragraphs respecting multi-column layouts.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import fitz
import structlog

from findocs.ingestion.models import TextBlock

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class TextExtractor:
    """Extracts structured text blocks from a single PDF page using PyMuPDF.

    Detects headings based on relative font size and reconstructs
    logical paragraphs from raw block coordinates.
    """

    HEADING_SIZE_RATIO: float = 1.2
    """A block's font size must exceed body_avg * this ratio to be a heading."""

    PARAGRAPH_GAP_RATIO: float = 1.5
    """Vertical gap between blocks must exceed line_height * this ratio to start a new paragraph."""

    def extract_blocks(self, page: fitz.Page) -> list[TextBlock]:
        """Extract text blocks with layout metadata from a single page.

        Uses ``page.get_text("dict")`` to retrieve blocks with font info
        and bounding-box coordinates.  Heading detection compares each
        block's dominant font size against the average body font size.

        Args:
            page: A PyMuPDF ``fitz.Page`` object.

        Returns:
            Ordered list of ``TextBlock`` objects with content, font size,
            heading flag, and bbox.
        """
        page_num: int = page.number + 1  # 1-indexed for user-facing data
        page_dict: dict = page.get_text("dict")
        raw_blocks: list = page_dict.get("blocks", [])

        # ------------------------------------------------------------------
        # First pass: collect every text span with its font info
        # ------------------------------------------------------------------
        span_records: list[dict] = []

        for block in raw_blocks:
            if block.get("type") != 0:
                # type 0 == text block; skip images (type 1)
                continue

            block_bbox = tuple(block["bbox"])
            lines = block.get("lines", [])

            for line in lines:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    span_records.append(
                        {
                            "text": text,
                            "font_size": span.get("size", 0.0),
                            "bold": "bold" in span.get("font", "").lower(),
                            "bbox": block_bbox,
                        }
                    )

        if not span_records:
            logger.debug("text_extractor.no_spans", page_num=page_num)
            return []

        # ------------------------------------------------------------------
        # Compute body-average font size (median is more robust than mean)
        # ------------------------------------------------------------------
        all_sizes = [s["font_size"] for s in span_records if s["font_size"] > 0]
        if all_sizes:
            body_avg_size: float = statistics.median(all_sizes)
        else:
            body_avg_size = 12.0  # safe fallback

        heading_threshold = body_avg_size * self.HEADING_SIZE_RATIO

        # ------------------------------------------------------------------
        # Second pass: aggregate spans back into per-block TextBlock objects
        # ------------------------------------------------------------------
        block_map: dict[tuple[float, float, float, float], list[dict]] = {}
        for rec in span_records:
            block_map.setdefault(rec["bbox"], []).append(rec)

        text_blocks: list[TextBlock] = []

        for bbox, spans in block_map.items():
            content = " ".join(s["text"] for s in spans)
            dominant_size = statistics.median([s["font_size"] for s in spans]) if spans else 0.0
            any_bold = any(s["bold"] for s in spans)
            is_heading = dominant_size >= heading_threshold or (any_bold and dominant_size > body_avg_size)

            text_blocks.append(
                TextBlock(
                    content=content,
                    is_heading=is_heading,
                    font_size=round(dominant_size, 2),
                    bbox=bbox,
                    page_num=page_num,
                )
            )

        logger.debug(
            "text_extractor.blocks_extracted",
            page_num=page_num,
            block_count=len(text_blocks),
            heading_count=sum(1 for b in text_blocks if b.is_heading),
        )
        return text_blocks

    # ------------------------------------------------------------------
    # Paragraph reconstruction
    # ------------------------------------------------------------------

    def reconstruct_paragraphs(self, blocks: list[TextBlock]) -> list[str]:
        """Merge adjacent ``TextBlock`` objects into logical paragraphs.

        Rules
        -----
        * Blocks are first sorted by x-coordinate then y-coordinate to
          handle multi-column layouts (left column before right).
        * A new paragraph starts when:
          - The block is a heading, **or**
          - The vertical gap from the previous block exceeds
            ``PARAGRAPH_GAP_RATIO`` times the estimated line height.
        * Heading blocks are prefixed with ``## `` (Markdown-style).

        Args:
            blocks: List of ``TextBlock`` objects (typically from one page).

        Returns:
            List of paragraph strings ready for downstream chunking.
        """
        if not blocks:
            return []

        # Sort by column (x0) then by row (y0) to handle multi-column PDFs.
        sorted_blocks = sorted(blocks, key=lambda b: (round(b.bbox[0], -1), b.bbox[1]))

        # Estimate line height from median font size
        median_font = statistics.median([b.font_size for b in sorted_blocks]) if sorted_blocks else 12.0
        line_height = median_font * 1.2  # typical line height ≈ 1.2x font size

        paragraphs: list[str] = []
        current_parts: list[str] = []

        prev_y1: float | None = None

        for block in sorted_blocks:
            y0 = block.bbox[1]

            start_new = False

            if block.is_heading:
                start_new = True
            elif prev_y1 is not None:
                gap = y0 - prev_y1
                if gap > line_height * self.PARAGRAPH_GAP_RATIO:
                    start_new = True

            if start_new and current_parts:
                paragraphs.append(" ".join(current_parts))
                current_parts = []

            if block.is_heading:
                current_parts.append(f"## {block.content}\n")
            else:
                current_parts.append(block.content)

            prev_y1 = block.bbox[3]

        # Flush remaining
        if current_parts:
            paragraphs.append(" ".join(current_parts))

        logger.debug(
            "text_extractor.paragraphs_reconstructed",
            paragraph_count=len(paragraphs),
        )
        return paragraphs
