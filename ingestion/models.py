"""Pydantic models for parsed financial documents."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """A block of text extracted from a PDF page with layout metadata."""

    content: str
    is_heading: bool
    font_size: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    page_num: int


class TableBlock(BaseModel):
    """A table extracted from a PDF page, converted to Markdown."""

    markdown_content: str
    caption: str | None = None
    page_num: int
    bbox: tuple[float, float, float, float]
    row_count: int
    col_count: int
    numerical_summary: dict[str, object] | None = None


class ChartBlock(BaseModel):
    """A chart/figure description extracted via vision LLM."""

    description: str
    chart_type: str | None = None
    page_num: int
    bbox: tuple[float, float, float, float]
    extraction_cost_usd: float


class ParsedBlock(BaseModel):
    """Unified block representation after parsing, ready for chunking."""

    content: str
    block_type: Literal["text", "table", "chart"]
    page_num: int
    metadata: dict[str, object] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Complete parsed document containing all extracted blocks."""

    source_path: Path
    doc_type: Literal["rbi_circular", "sebi_factsheet", "nse_annual_report"]
    title: str
    date: datetime | None = None
    blocks: list[ParsedBlock]
    total_pages: int
    parsing_duration_seconds: float


class DownloadedDocument(BaseModel):
    """Metadata for a downloaded document before parsing."""

    local_path: Path
    url: str
    title: str
    date: datetime | None = None
    doc_type: str
