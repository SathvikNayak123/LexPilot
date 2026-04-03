from pydantic import BaseModel
from typing import Literal, Optional
from datetime import date


class ParsedBlock(BaseModel):
    """A single parsed content block from a PDF page."""
    block_type: Literal["text", "table", "heading"]
    content: str
    page_number: int
    font_size: Optional[float] = None
    is_heading: bool = False
    heading_level: Optional[int] = None  # 1=title, 2=section, 3=subsection
    bbox: Optional[tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    # Table-specific
    row_count: Optional[int] = None
    col_count: Optional[int] = None


class ParsedDocument(BaseModel):
    """Complete parsed document with metadata."""
    document_id: str
    title: str
    doc_type: Literal["judgment", "contract", "statute", "policy"]
    source: Optional[str] = None
    citation: Optional[str] = None           # Primary citation, e.g. "(2017) 10 SCC 1"
    citation_aliases: list[str] = []         # Same judgment in other reporters, e.g. ["AIR 2017 SC 1", "1976 SCR 172"]
    court: Optional[str] = None  # For judgments
    date: Optional[date] = None
    blocks: list[ParsedBlock]
    total_pages: int
    parse_duration_ms: int
    metadata: dict = {}
