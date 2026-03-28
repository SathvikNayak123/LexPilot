import fitz  # PyMuPDF
import pdfplumber
import time
from pathlib import Path

import structlog

from ingestion.models import ParsedBlock, ParsedDocument

logger = structlog.get_logger()


class PDFParser:
    """Extracts text and tables from legal PDFs.

    Strategy:
    - PyMuPDF (fitz) for text blocks with font metadata (fast, high fidelity)
    - pdfplumber for table detection and extraction (best-in-class for tables)
    - Heading detection via font size heuristics
    """

    # Font size thresholds for heading detection (legal docs)
    HEADING_THRESHOLDS = {
        1: 18.0,  # Title
        2: 14.0,  # Section
        3: 12.0,  # Subsection
    }

    def parse(self, pdf_path: str | Path, doc_type: str = "judgment",
              source: str = None, court: str = None) -> ParsedDocument:
        """Parse a PDF and return a structured ParsedDocument."""
        start = time.time()
        pdf_path = Path(pdf_path)

        blocks = []
        total_pages = 0

        # Pass 1: Text extraction with PyMuPDF
        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            for page_num, page in enumerate(doc, 1):
                text_blocks = self._extract_text_blocks(page, page_num)
                blocks.extend(text_blocks)

        # Pass 2: Table extraction with pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                table_blocks = self._extract_tables(page, page_num)
                blocks.extend(table_blocks)

        # Sort blocks by page, then vertical position
        blocks.sort(key=lambda b: (b.page_number, b.bbox[1] if b.bbox else 0))

        # Detect title
        title = self._detect_title(blocks) or pdf_path.stem

        duration = int((time.time() - start) * 1000)
        logger.info("pdf_parsed", path=str(pdf_path), pages=total_pages,
                     blocks=len(blocks), duration_ms=duration)

        return ParsedDocument(
            document_id=f"doc_{pdf_path.stem}_{int(time.time())}",
            title=title,
            doc_type=doc_type,
            source=source,
            court=court,
            blocks=blocks,
            total_pages=total_pages,
            parse_duration_ms=duration,
        )

    def _extract_text_blocks(self, page, page_num: int) -> list[ParsedBlock]:
        """Extract text blocks with font metadata from a PyMuPDF page."""
        blocks = []
        raw_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)

        for block in raw_blocks.get("blocks", []):
            if block.get("type") != 0:  # text block
                continue

            text_parts = []
            max_font_size = 0

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_parts.append(span["text"])
                    max_font_size = max(max_font_size, span["size"])

            text = " ".join(text_parts).strip()
            if not text or len(text) < 3:
                continue

            # Detect heading level
            heading_level = None
            is_heading = False
            for level, threshold in sorted(self.HEADING_THRESHOLDS.items()):
                if max_font_size >= threshold:
                    heading_level = level
                    is_heading = True

            bbox = (block["bbox"][0], block["bbox"][1],
                    block["bbox"][2], block["bbox"][3])

            blocks.append(ParsedBlock(
                block_type="heading" if is_heading else "text",
                content=text,
                page_number=page_num,
                font_size=max_font_size,
                is_heading=is_heading,
                heading_level=heading_level,
                bbox=bbox,
            ))

        return blocks

    def _extract_tables(self, page, page_num: int) -> list[ParsedBlock]:
        """Extract tables from a pdfplumber page and convert to Markdown."""
        blocks = []
        tables = page.find_tables()

        for table in tables:
            extracted = table.extract()
            if not extracted or len(extracted) < 2:
                continue

            md_table = self._table_to_markdown(extracted)
            bbox = table.bbox if hasattr(table, 'bbox') else None

            blocks.append(ParsedBlock(
                block_type="table",
                content=md_table,
                page_number=page_num,
                bbox=bbox,
                row_count=len(extracted),
                col_count=len(extracted[0]) if extracted else 0,
            ))

        return blocks

    def _table_to_markdown(self, table_data: list[list]) -> str:
        """Convert a 2D table to Markdown format."""
        if not table_data:
            return ""

        headers = table_data[0]
        header_row = "| " + " | ".join(str(h or "") for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"

        rows = []
        for row in table_data[1:]:
            rows.append("| " + " | ".join(str(c or "") for c in row) + " |")

        return "\n".join([header_row, separator] + rows)

    def _detect_title(self, blocks: list[ParsedBlock]) -> str | None:
        """Detect document title from first heading block."""
        for block in blocks[:10]:
            if block.is_heading and block.heading_level == 1:
                return block.content[:200]
        return None
