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
    court: Optional[str] = None  # For judgments
    date: Optional[date] = None
    blocks: list[ParsedBlock]
    total_pages: int
    parse_duration_ms: int
    metadata: dict = {}
