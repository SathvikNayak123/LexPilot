import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from datetime import datetime
import uuid
import structlog
import tiktoken

from config.config import settings

logger = structlog.get_logger()


class ResearchSession(BaseModel):
    id: str
    user_id: str
    name: str
    summary: dict = {}
    precedents_found: list[str] = []
    clauses_analyzed: list[dict] = []
    compliance_findings: list[dict] = []
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ResearchMemory:
    """Cross-session research memory.

    A lawyer investigating a case might have 5 conversations over a week.
    ResearchSession accumulates findings across conversations and injects
    a compressed summary into the system prompt on each new interaction.
    """

    def __init__(self):
        self.engine = create_async_engine(settings.postgres_url)

    async def create_session(self, user_id: str, name: str) -> ResearchSession:
        """Create a new research session."""
        session_id = f"rs_{uuid.uuid4().hex[:12]}"

        async with AsyncSession(self.engine) as db:
            await db.execute(
                text("""
                    INSERT INTO research_sessions (id, user_id, name)
                    VALUES (:id, :uid, :name)
                """),
                {"id": session_id, "uid": user_id, "name": name},
            )
            await db.commit()

        return ResearchSession(id=session_id, user_id=user_id, name=name)

    async def get_session(self, session_id: str) -> ResearchSession | None:
        async with AsyncSession(self.engine) as db:
            result = await db.execute(
                text("SELECT * FROM research_sessions WHERE id = :id"),
                {"id": session_id},
            )
            row = result.mappings().first()
            if row:
                return ResearchSession(**dict(row))
            return None

    async def list_sessions(self, user_id: str) -> list[ResearchSession]:
        async with AsyncSession(self.engine) as db:
            result = await db.execute(
                text(
                    "SELECT * FROM research_sessions WHERE user_id = :uid ORDER BY updated_at DESC"
                ),
                {"uid": user_id},
            )
            return [ResearchSession(**dict(row)) for row in result.mappings().all()]

    async def update_session(
        self,
        session_id: str,
        precedents: list[str] | None = None,
        clauses: list[dict] | None = None,
        compliance: list[dict] | None = None,
        summary_update: dict | None = None,
    ):
        """Append new findings to a research session."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        updates = {}
        if precedents:
            merged = list(set(session.precedents_found + precedents))
            updates["precedents_found"] = json.dumps(merged)
        if clauses:
            merged = session.clauses_analyzed + clauses
            updates["clauses_analyzed"] = json.dumps(merged)
        if compliance:
            merged = session.compliance_findings + compliance
            updates["compliance_findings"] = json.dumps(merged)
        if summary_update:
            merged = {**session.summary, **summary_update}
            updates["summary"] = json.dumps(merged)

        if updates:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            updates["id"] = session_id
            async with AsyncSession(self.engine) as db:
                await db.execute(
                    text(
                        f"UPDATE research_sessions SET {set_clause}, updated_at = NOW() WHERE id = :id"
                    ),
                    updates,
                )
                await db.commit()

    def build_context_injection(
        self, session: ResearchSession, max_tokens: int = 500
    ) -> str:
        """Build a compressed context string for injection into the system prompt."""
        parts = [f"Research Session: {session.name}"]

        if session.precedents_found:
            parts.append(
                f"Precedents identified: {', '.join(session.precedents_found[:10])}"
            )

        if session.compliance_findings:
            gap_count = len(session.compliance_findings)
            parts.append(f"Compliance gaps found: {gap_count}")

        if session.summary:
            parts.append(f"Key findings: {json.dumps(session.summary)}")

        context = "\n".join(parts)
        # Truncate to max tokens if needed
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(context)
        if len(tokens) > max_tokens:
            context = enc.decode(tokens[:max_tokens]) + "\n[... truncated]"

        return context
