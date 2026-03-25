"""pdfplumber-based table extraction to structured Markdown.

Opens the PDF with *pdfplumber*, extracts tables for a given page,
post-processes cells, detects headers, and converts to GitHub-style
Markdown.  Numeric-heavy tables also get a statistical summary.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber
import structlog

from findocs.ingestion.models import TableBlock

logger = structlog.get_logger(__name__)

# Regex that matches a number (int, float, negative, percent, currency-prefixed)
_NUMERIC_RE = re.compile(
    r"^[\s$\u20b9\u20ac\u00a3]*-?\d[\d,]*\.?\d*%?\s*$"
)


class TableExtractor:
    """Extract tables from a PDF page and convert them to Markdown.

    Uses *pdfplumber* for table detection and cell extraction.  Each
    extracted table is returned as a ``TableBlock`` with Markdown content,
    optional caption, and an optional numerical summary.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_tables(self, pdf_path: Path, page_num: int) -> list[TableBlock]:
        """Extract all tables from a single page of a PDF.

        Args:
            pdf_path: Filesystem path to the PDF file.
            page_num: **1-indexed** page number to process.

        Returns:
            List of ``TableBlock`` objects, one per detected table.
        """
        table_blocks: list[TableBlock] = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if page_num < 1 or page_num > len(pdf.pages):
                    logger.warning(
                        "table_extractor.invalid_page",
                        pdf_path=str(pdf_path),
                        page_num=page_num,
                        total_pages=len(pdf.pages),
                    )
                    return []

                page = pdf.pages[page_num - 1]  # pdfplumber is 0-indexed
                page_text = page.extract_text() or ""
                tables = page.find_tables()

                for idx, table_obj in enumerate(tables):
                    raw_table: list[list[str | None]] = table_obj.extract()
                    if not raw_table:
                        continue

                    cleaned = self._clean_table(raw_table)
                    if not cleaned:
                        continue

                    headers, body = self._detect_header(cleaned)
                    caption = self._detect_caption(page_text, table_obj.bbox)
                    markdown = self.table_to_markdown(body, headers)
                    numerical_summary = self.extract_numerical_summary(cleaned)

                    bbox = tuple(table_obj.bbox)  # (x0, y0, x1, y1)

                    block = TableBlock(
                        markdown_content=markdown,
                        caption=caption,
                        page_num=page_num,
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        row_count=len(body),
                        col_count=len(headers),
                        numerical_summary=numerical_summary,
                    )
                    table_blocks.append(block)

                    logger.debug(
                        "table_extractor.table_extracted",
                        pdf_path=str(pdf_path),
                        page_num=page_num,
                        table_index=idx,
                        rows=block.row_count,
                        cols=block.col_count,
                    )

        except Exception:
            logger.exception(
                "table_extractor.extraction_error",
                pdf_path=str(pdf_path),
                page_num=page_num,
            )

        return table_blocks

    # ------------------------------------------------------------------
    # Markdown conversion
    # ------------------------------------------------------------------

    def table_to_markdown(self, table: list[list[str]], headers: list[str]) -> str:
        """Convert a table (list of rows) into a GitHub-style Markdown table.

        Args:
            table: Body rows — each row is a list of cell strings.
            headers: Column header strings.

        Returns:
            A Markdown-formatted table string.
        """
        col_count = len(headers)

        # Ensure every row matches header width
        normalised_rows: list[list[str]] = []
        for row in table:
            padded = list(row) + [""] * max(0, col_count - len(row))
            normalised_rows.append(padded[:col_count])

        # Build lines
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"

        body_lines: list[str] = []
        for row in normalised_rows:
            body_lines.append("| " + " | ".join(row) + " |")

        return "\n".join([header_line, sep_line, *body_lines])

    # ------------------------------------------------------------------
    # Numerical summary
    # ------------------------------------------------------------------

    def extract_numerical_summary(self, table: list[list[str]]) -> dict[str, Any] | None:
        """Generate a numerical summary if >50% of cells are numeric.

        For qualifying tables the summary contains:
        * ``min`` / ``max`` — global extremes
        * ``sum`` — grand total of all numeric cells
        * ``column_totals`` — per-column sums keyed by column index

        Args:
            table: Full table including header row (list of rows).

        Returns:
            Summary dict, or ``None`` if the table is not numeric-heavy.
        """
        if not table:
            return None

        total_cells = 0
        numeric_cells = 0
        all_values: list[float] = []
        # column index -> list of values
        col_values: dict[int, list[float]] = {}

        for row in table:
            for col_idx, cell in enumerate(row):
                total_cells += 1
                value = self._parse_numeric(cell)
                if value is not None:
                    numeric_cells += 1
                    all_values.append(value)
                    col_values.setdefault(col_idx, []).append(value)

        if total_cells == 0 or (numeric_cells / total_cells) <= 0.5:
            return None

        column_totals = {str(k): round(sum(v), 4) for k, v in sorted(col_values.items())}

        summary: dict[str, Any] = {
            "min": round(min(all_values), 4),
            "max": round(max(all_values), 4),
            "sum": round(sum(all_values), 4),
            "column_totals": column_totals,
        }

        logger.debug(
            "table_extractor.numerical_summary",
            numeric_pct=round(numeric_cells / total_cells * 100, 1),
            summary_keys=list(summary.keys()),
        )
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_table(raw: list[list[str | None]]) -> list[list[str]]:
        """Strip whitespace and replace ``None`` cells with empty strings.

        Drops rows that are entirely empty after cleaning.

        Args:
            raw: Raw table rows from pdfplumber.

        Returns:
            Cleaned table rows.
        """
        cleaned: list[list[str]] = []
        for row in raw:
            cells = [(cell.strip() if cell else "") for cell in row]
            if any(cells):
                cleaned.append(cells)
        return cleaned

    @staticmethod
    def _detect_header(table: list[list[str]]) -> tuple[list[str], list[list[str]]]:
        """Split the first row as headers, or generate synthetic ones.

        The first row is used as the header if at least half its cells are
        non-numeric.  Otherwise, synthetic ``Column_1 ... Column_N``
        headers are created and the full table is returned as the body.

        Args:
            table: Cleaned table rows (at least one row).

        Returns:
            A ``(headers, body_rows)`` tuple.
        """
        if not table:
            return [], []

        first_row = table[0]
        non_numeric = sum(1 for c in first_row if c and not _NUMERIC_RE.match(c))

        if non_numeric >= len(first_row) / 2:
            # First row looks like a header
            headers = first_row
            body = table[1:]
        else:
            # Generate synthetic headers
            col_count = max(len(r) for r in table) if table else 0
            headers = [f"Column_{i + 1}" for i in range(col_count)]
            body = table

        return headers, body

    @staticmethod
    def _detect_caption(page_text: str, bbox: tuple[float, ...]) -> str | None:
        """Try to find a caption for the table.

        Heuristic: if any line in the page text that ends with ``:``
        appears *above* the table's top edge, use it as the caption.

        Args:
            page_text: Full extracted text of the page.
            bbox: Table bounding box ``(x0, y0, x1, y1)``.

        Returns:
            Caption string or ``None``.
        """
        lines = page_text.split("\n")
        # Walk backwards through lines looking for one ending with ":"
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.endswith(":"):
                return stripped.rstrip(":")
        return None

    @staticmethod
    def _parse_numeric(cell: str) -> float | None:
        """Attempt to parse a cell string as a float.

        Strips currency symbols, commas, percentage signs, and whitespace
        before conversion.

        Args:
            cell: Raw cell string.

        Returns:
            Parsed float, or ``None`` if parsing fails.
        """
        if not cell or not cell.strip():
            return None
        cleaned = cell.strip()
        # Remove common currency and formatting characters
        cleaned = re.sub(r"[$\u20b9\u20ac\u00a3,%]", "", cleaned)
        cleaned = cleaned.strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
