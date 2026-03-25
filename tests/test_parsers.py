"""Tests for ingestion parsers: TextExtractor, TableExtractor, ChartExtractor, PDFParser.

All external services (OpenAI Vision) are mocked.  PDF fixtures are created
dynamically with fpdf2 so the test suite has zero external dependencies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import pytest
from fpdf import FPDF

from findocs.ingestion.models import ChartBlock, ParsedBlock, TableBlock, TextBlock
from findocs.ingestion.parsers.chart_extractor import ChartExtractor
from findocs.ingestion.parsers.pdf_parser import PDFParser
from findocs.ingestion.parsers.table_extractor import TableExtractor
from findocs.ingestion.parsers.text_extractor import TextExtractor


# ============================================================================
# TextExtractor tests
# ============================================================================


class TestTextExtractor:
    """Tests for the TextExtractor class."""

    def _create_pdf_with_text(self, tmp_path: Path, text: str, font_size: float = 11) -> Path:
        """Helper to create a single-page PDF with given text and font size."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=font_size)
        pdf.multi_cell(0, 7, text=text)
        output_path = tmp_path / "text_test.pdf"
        pdf.output(str(output_path))
        return output_path

    def test_text_extractor_extracts_blocks(self, tmp_path: Path) -> None:
        """Create a PDF with known text, run TextExtractor, verify blocks contain expected text."""
        expected_text = (
            "The Reserve Bank of India has decided to maintain the repo rate "
            "at 6.50 percent for the current fiscal year."
        )
        pdf_path = self._create_pdf_with_text(tmp_path, expected_text)

        doc = fitz.open(str(pdf_path))
        page = doc[0]

        extractor = TextExtractor()
        blocks = extractor.extract_blocks(page)
        doc.close()

        assert len(blocks) > 0, "TextExtractor should return at least one block"
        all_content = " ".join(b.content for b in blocks)
        # The key phrase should be present in the extracted text
        assert "repo rate" in all_content.lower(), (
            f"Expected 'repo rate' in extracted text, got: {all_content}"
        )
        assert "6.50" in all_content, (
            f"Expected '6.50' in extracted text, got: {all_content}"
        )

        # All blocks should be TextBlock instances
        for block in blocks:
            assert isinstance(block, TextBlock)
            assert block.page_num >= 1
            assert block.font_size > 0
            assert len(block.bbox) == 4

    def test_text_extractor_detects_headings(self, tmp_path: Path) -> None:
        """Create a PDF with large-font heading + normal body, verify heading detection."""
        pdf = FPDF()
        pdf.add_page()

        # Write heading in large font
        pdf.set_font("Helvetica", "B", size=22)
        pdf.cell(0, 12, text="Monetary Policy Statement", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Write body in normal font
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(
            0,
            6,
            text=(
                "The Monetary Policy Committee met on February 7, 2025 and decided to "
                "keep the policy repo rate unchanged at 6.50 percent. This is the "
                "continuation of the current monetary stance."
            ),
        )

        output_path = tmp_path / "heading_test.pdf"
        pdf.output(str(output_path))

        doc = fitz.open(str(output_path))
        page = doc[0]

        extractor = TextExtractor()
        blocks = extractor.extract_blocks(page)
        doc.close()

        assert len(blocks) >= 1, "Should extract at least one block"

        # Check that at least one block is detected as a heading
        headings = [b for b in blocks if b.is_heading]
        non_headings = [b for b in blocks if not b.is_heading]

        # The large-font block should be a heading
        assert len(headings) >= 1, (
            f"Expected at least one heading block, got {len(headings)}. "
            f"Block sizes: {[(b.content[:30], b.font_size, b.is_heading) for b in blocks]}"
        )

        # If there are non-heading blocks, they should have a smaller font
        if headings and non_headings:
            max_heading_size = max(h.font_size for h in headings)
            min_body_size = min(b.font_size for b in non_headings)
            assert max_heading_size > min_body_size, (
                f"Heading font ({max_heading_size}) should be larger than body font ({min_body_size})"
            )


# ============================================================================
# TableExtractor tests
# ============================================================================


class TestTableExtractor:
    """Tests for the TableExtractor class."""

    def test_table_to_markdown_format(self) -> None:
        """Test table_to_markdown produces valid GitHub-style markdown with | separators."""
        extractor = TableExtractor()

        headers = ["Metric", "Value", "Change"]
        body = [
            ["Repo Rate", "6.50%", "-25 bps"],
            ["CRR", "4.50%", "No change"],
            ["Reverse Repo", "3.35%", "No change"],
        ]

        markdown = extractor.table_to_markdown(body, headers)

        # Verify the markdown structure
        lines = markdown.strip().split("\n")
        assert len(lines) == 5, f"Expected 5 lines (header + separator + 3 body), got {len(lines)}"

        # Header line
        assert lines[0].startswith("| ")
        assert lines[0].endswith(" |")
        assert "Metric" in lines[0]
        assert "Value" in lines[0]
        assert "Change" in lines[0]

        # Separator line
        assert "---" in lines[1]
        assert lines[1].startswith("| ")

        # Body lines
        assert "Repo Rate" in lines[2]
        assert "6.50%" in lines[2]
        assert "CRR" in lines[3]
        assert "Reverse Repo" in lines[4]

        # All lines should have the correct number of | separators
        for line in lines:
            pipe_count = line.count("|")
            assert pipe_count == len(headers) + 1, (
                f"Expected {len(headers) + 1} pipe chars, got {pipe_count} in: {line}"
            )

    def test_table_to_markdown_header_detection(self) -> None:
        """Test auto-header detection when first row has short non-numeric strings."""
        extractor = TableExtractor()

        # A table where first row clearly looks like a header (short text strings)
        table = [
            ["Fund Name", "AUM (Cr)", "1Y Return"],
            ["SBI Bluechip", "45000", "22.5%"],
            ["HDFC Top 100", "32000", "18.3%"],
        ]

        headers, body = extractor._detect_header(table)

        # First row should be detected as header
        assert headers == ["Fund Name", "AUM (Cr)", "1Y Return"]
        assert len(body) == 2
        assert body[0][0] == "SBI Bluechip"

    def test_table_to_markdown_header_detection_numeric_first_row(self) -> None:
        """When first row is all numeric, synthetic headers should be generated."""
        extractor = TableExtractor()

        table = [
            ["100", "200", "300"],
            ["400", "500", "600"],
        ]

        headers, body = extractor._detect_header(table)

        # Synthetic headers expected
        assert headers[0].startswith("Column_")
        assert len(headers) == 3
        # Full table should be the body
        assert len(body) == 2

    def test_table_extractor_numerical_summary(self) -> None:
        """Test numerical summary extraction (min, max, sum) on a numeric-heavy table."""
        extractor = TableExtractor()

        # Table where >50% of cells are numeric
        table = [
            ["Q1", "100", "200"],
            ["Q2", "150", "250"],
            ["Q3", "200", "300"],
            ["Q4", "250", "350"],
        ]

        summary = extractor.extract_numerical_summary(table)

        assert summary is not None, "Numeric-heavy table should produce a summary"
        assert "min" in summary
        assert "max" in summary
        assert "sum" in summary
        assert "column_totals" in summary

        assert summary["min"] == 100.0
        assert summary["max"] == 350.0
        # Sum of all numbers: 100+200+150+250+200+300+250+350 = 1800
        assert summary["sum"] == 1800.0

    def test_table_extractor_numerical_summary_non_numeric(self) -> None:
        """Tables that are mostly text should return None for numerical_summary."""
        extractor = TableExtractor()

        table = [
            ["Policy", "Active", "RBI"],
            ["Regulation", "Pending", "SEBI"],
            ["Guideline", "Draft", "NSE"],
        ]

        summary = extractor.extract_numerical_summary(table)
        assert summary is None, "Non-numeric table should return None"


# ============================================================================
# ChartExtractor tests
# ============================================================================


class TestChartExtractor:
    """Tests for the ChartExtractor class."""

    @pytest.mark.asyncio
    async def test_chart_extractor_returns_chart_block(self, mock_openai_client: MagicMock) -> None:
        """Mock OpenAI Vision, verify ChartBlock has description, chart_type, page_num, bbox, extraction_cost_usd."""

        with patch(
            "findocs.ingestion.parsers.chart_extractor.AsyncOpenAI",
            return_value=mock_openai_client,
        ):
            extractor = ChartExtractor(openai_api_key="test-key")
            # Replace the internal client with our mock
            extractor._client = mock_openai_client

            # Create a mock fitz.Page
            mock_page = MagicMock()
            mock_page.number = 0  # 0-indexed, so page_num = 1
            mock_pixmap = MagicMock()
            mock_pixmap.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            mock_page.get_pixmap.return_value = mock_pixmap

            # Create a mock Rect
            mock_rect = MagicMock()
            mock_rect.x0 = 50.0
            mock_rect.y0 = 100.0
            mock_rect.x1 = 400.0
            mock_rect.y1 = 300.0

            with patch("findocs.ingestion.parsers.chart_extractor.fitz.Rect", return_value=mock_rect):
                result = await extractor.extract_chart_description(mock_page, mock_rect)

        assert result is not None, "ChartExtractor should return a ChartBlock"
        assert isinstance(result, ChartBlock)

        # Verify required fields
        assert result.description, "ChartBlock should have a non-empty description"
        assert result.chart_type is not None, "ChartBlock should have chart_type"
        assert result.chart_type == "bar", f"Expected chart_type 'bar', got '{result.chart_type}'"
        assert result.page_num == 1, f"Expected page_num 1, got {result.page_num}"
        assert len(result.bbox) == 4, "bbox should have 4 elements"
        assert result.extraction_cost_usd >= 0, "extraction_cost_usd should be non-negative"


# ============================================================================
# PDFParser tests
# ============================================================================


class TestPDFParser:
    """Tests for the PDFParser class."""

    def test_pdf_parser_detect_content_types(self, tmp_pdf: Path) -> None:
        """Test page content type detection heuristics on a real PDF."""
        parser = PDFParser(openai_api_key="test-key")

        doc = fitz.open(str(tmp_pdf))
        page = doc[0]

        content_types = parser.detect_page_content_types(page)
        doc.close()

        # Our test PDF has text content, so "text" should be detected
        assert "text" in content_types, (
            f"Expected 'text' in content types, got: {content_types}"
        )

    def test_pdf_parser_detect_content_types_empty_page(self, tmp_path: Path) -> None:
        """An empty page should return an empty or minimal set of content types."""
        # Create a PDF with an empty page
        pdf = FPDF()
        pdf.add_page()
        output_path = tmp_path / "empty_page.pdf"
        pdf.output(str(output_path))

        parser = PDFParser(openai_api_key="test-key")
        doc = fitz.open(str(output_path))
        page = doc[0]

        content_types = parser.detect_page_content_types(page)
        doc.close()

        # An empty page should NOT have "text" (less than 100 chars)
        assert "text" not in content_types, (
            f"Empty page should not detect 'text', got: {content_types}"
        )
