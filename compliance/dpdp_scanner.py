import json
import instructor
from anthropic import Anthropic
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
import structlog

from config.config import settings
from compliance.dpdp_sections import DPDP_SECTIONS
from compliance.models import ComplianceGap, ComplianceReport

logger = structlog.get_logger()


class DPDPScanner:
    """Exhaustive clause-level DPDP Act 2023 compliance scanner.

    Unlike RAG (which retrieves top-k relevant chunks), this scanner
    processes EVERY clause in the document against ALL DPDP sections.
    This is necessary because a compliance audit must not miss gaps.

    Pipeline: map (per-clause tagging, Tier 2 model) -> reduce (synthesis, Tier 3 model)
    """

    def __init__(self):
        self.engine = create_async_engine(settings.postgres_url)
        self.valid_sections = set(DPDP_SECTIONS.keys())

        # Tier 2 for map step (cheap, focused extraction)
        self.tag_client = instructor.from_litellm(
            model="groq/llama-3.3-70b-versatile",
        )
        # Tier 3 for reduce step (synthesis needs strong reasoning)
        self.synthesis_client = instructor.from_anthropic(
            Anthropic(api_key=settings.anthropic_api_key),
        )

    async def scan(self, document_id: str) -> ComplianceReport:
        """Run full compliance scan on a document."""
        # Load all parent chunks for this document
        chunks = await self._load_document_chunks(document_id)
        if not chunks:
            raise ValueError(f"No chunks found for document {document_id}")

        # MAP: Tag each clause against DPDP sections (Tier 2, parallel-ish)
        all_gaps = []
        for chunk in chunks:
            gaps = await self._tag_clause(chunk)
            all_gaps.extend(gaps)

        # Validate DPDP sections (reject hallucinated sections)
        validated_gaps = []
        for gap in all_gaps:
            if gap.dpdp_section in self.valid_sections:
                validated_gaps.append(gap)
            else:
                logger.warning("invalid_dpdp_section",
                               section=gap.dpdp_section, clause=gap.clause_text[:50])

        # REDUCE: Synthesize into report (Tier 3)
        report = await self._synthesize_report(document_id, chunks, validated_gaps)
        return report

    async def _load_document_chunks(self, document_id: str) -> list[dict]:
        """Load all parent chunks for a document from Postgres."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                text("SELECT id, content, metadata FROM parent_chunks WHERE document_id = :did ORDER BY id"),
                {"did": document_id},
            )
            return [
                {"id": row[0], "content": row[1], "metadata": json.loads(row[2]) if row[2] else {}}
                for row in result.fetchall()
            ]

    async def _tag_clause(self, chunk: dict) -> list[ComplianceGap]:
        """MAP step: Tag a single clause against DPDP sections. Uses Tier 2 model."""
        try:
            gaps = self.tag_client.chat.completions.create(
                model="groq/llama-3.3-70b-versatile",
                response_model=list[ComplianceGap],
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this contract clause for DPDP Act 2023 compliance gaps.

Valid DPDP sections: {list(self.valid_sections)}

Clause text:
{chunk['content']}

Return a list of compliance gaps found (empty list if fully compliant).
Each gap must reference a valid DPDP section from the list above.
Only flag genuine gaps - do not over-report.""",
                }],
                max_retries=2,
            )
            return gaps
        except Exception as e:
            logger.warning("clause_tagging_failed", chunk_id=chunk["id"], error=str(e))
            return []

    async def _synthesize_report(self, document_id: str, chunks: list,
                                  gaps: list[ComplianceGap]) -> ComplianceReport:
        """REDUCE step: Synthesize gaps into a coherent report. Uses Tier 3 model."""
        # Determine overall risk
        critical_count = sum(1 for g in gaps if g.risk_level == "critical")
        high_count = sum(1 for g in gaps if g.risk_level == "high")

        if critical_count > 0:
            overall = "non_compliant"
        elif high_count > 2:
            overall = "major_gaps"
        elif gaps:
            overall = "minor_gaps"
        else:
            overall = "compliant"

        # Generate executive summary using Tier 3
        summary_prompt = f"""Summarize DPDP compliance audit findings:
Total clauses analyzed: {len(chunks)}
Total gaps found: {len(gaps)}
Critical: {critical_count}, High: {high_count}
Overall risk: {overall}

Key gaps:
{json.dumps([g.model_dump() for g in gaps[:10]], indent=2)}

Write a 3-4 sentence executive summary for a legal reviewer."""

        response = self.synthesis_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = response.content[0].text

        return ComplianceReport(
            document_id=document_id,
            overall_risk=overall,
            total_clauses_analyzed=len(chunks),
            gaps=gaps,
            sections_analyzed=list(set(g.dpdp_section for g in gaps)),
            executive_summary=summary,
        )
