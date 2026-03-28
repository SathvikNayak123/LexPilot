from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
import json
from config.config import settings
from chunking.models import ParentChunk


class ParentChunkStore:
    """Stores parent chunks in PostgreSQL."""

    def __init__(self):
        self.engine = create_async_engine(settings.postgres_url)

    async def store(self, parents: list[ParentChunk]):
        async with AsyncSession(self.engine) as session:
            for p in parents:
                await session.execute(
                    text("""
                        INSERT INTO parent_chunks (id, document_id, content, metadata, token_count)
                        VALUES (:id, :did, :content, :meta, :tc)
                        ON CONFLICT (id) DO UPDATE SET content = :content, metadata = :meta
                    """),
                    {
                        "id": p.id, "did": p.document_id,
                        "content": p.content,
                        "meta": json.dumps(p.metadata.model_dump()),
                        "tc": p.token_count,
                    },
                )
            await session.commit()
