import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from config.config import get_db_engine
from chunking.models import ParentChunk


class ParentChunkStore:
    """Stores parent chunks in PostgreSQL."""

    async def store(self, parents: list[ParentChunk]):
        if not parents:
            return
        rows = [
            {
                "id": p.id,
                "did": p.document_id,
                "content": p.content,
                "meta": json.dumps(p.metadata.model_dump()),
                "tc": p.token_count,
            }
            for p in parents
        ]
        async with AsyncSession(get_db_engine()) as session:
            await session.execute(
                text("""
                    INSERT INTO parent_chunks (id, document_id, content, metadata, token_count)
                    VALUES (:id, :did, :content, :meta, :tc)
                    ON CONFLICT (id) DO UPDATE SET content = :content, metadata = :meta
                """),
                rows,
            )
            await session.commit()
