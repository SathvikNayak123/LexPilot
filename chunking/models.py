from pydantic import BaseModel
from typing import Optional


class ChunkMetadata(BaseModel):
    document_id: str
    doc_type: str  # "judgment", "contract", "statute", "policy"
    source: Optional[str] = None
    court: Optional[str] = None
    date: Optional[str] = None
    heading_context: str = ""  # Nearest heading above this chunk
    page_number: int = 0
    chunk_index: int = 0


class ChildChunk(BaseModel):
    """Small chunk (~128 tokens) indexed in Qdrant for retrieval precision."""
    id: str
    parent_id: str
    content: str
    token_count: int
    char_count: int
    metadata: ChunkMetadata


class ParentChunk(BaseModel):
    """Larger chunk (~512 tokens) stored in Postgres for LLM context."""
    id: str
    document_id: str
    content: str
    token_count: int
    char_count: int
    metadata: ChunkMetadata
    child_ids: list[str] = []
