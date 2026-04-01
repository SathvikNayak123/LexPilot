import json
import os
import instructor
import litellm
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
import structlog

from config.config import settings
from compliance.dpdp_sections import DPDP_SECTIONS
from compliance.models import ComplianceGap, ComplianceReport

logger = structlog.get_logger()

# Set OpenRouter API key for litellm via environment variable
os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key

# Suppress litellm's "Provider List" banner and verbose output
litellm.suppress_debug_info = True
litellm.set_verbose = False


class DPDPScanner:
    """Exhaustive clause-level DPDP Act 2023 compliance scanner.

    Unlike RAG (which retrieves top-k relevant chunks), this scanner
    processes EVERY clause in the document against ALL DPDP sections.
    This is necessary because a compliance audit must not miss gaps.

    Pipeline: map (per-clause tagging) -> reduce (synthesis)
    """

    # LiteLLM model string for OpenRouter (see https://docs.litellm.ai/docs/providers/openrouter)
    OPENROUTER_MODEL = "openrouter/google/gemini-3-flash-preview"

    def __init__(self):
        self.engine = create_async_engine(settings.postgres_url)
        self.valid_sections = set(DPDP_SECTIONS.keys())

        # Async instructor client wrapping litellm for structured outputs
        # Use JSON mode instead of tool calling — Gemini Flash is more reliable with it
        self.tag_client = instructor.from_litellm(
            litellm.acompletion, mode=instructor.Mode.JSON
        )

    async def scan(self, document_id: str, clauses: list[str] = None) -> ComplianceReport:
        """Run full compliance scan on a document.

        Args:
            document_id: Document ID to load chunks from Postgres.
            clauses: Optional list of clause texts to scan directly (bypasses DB).
        """
        if clauses:
            chunks = [{"id": f"{document_id}_c{i}", "content": c, "metadata": {}}
                      for i, c in enumerate(clauses)]
        else:
            # Load all parent chunks for this document
            chunks = await self._load_document_chunks(document_id)
            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")

        # MAP: Tag each clause against DPDP sections
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
        """MAP step: Tag a single clause against DPDP sections. Uses async litellm."""
        try:
            gaps = await self.tag_client.chat.completions.create(
                model=self.OPENROUTER_MODEL,
                response_model=list[ComplianceGap],
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this contract clause for DPDP Act 2023 compliance gaps.
Also flag MISSING obligations — if a required DPDP mechanism is absent from this clause, that is a gap.

Valid DPDP sections (you MUST use one of these exact strings):
{chr(10).join(f'- "{k}": {v["title"]} — {v["description"]}' for k, v in DPDP_SECTIONS.items())}

Clause text:
{chunk['content']}

For each compliance gap found, return a JSON object with these exact fields:
- "clause_text": the relevant excerpt from the clause (or "absent" if the mechanism is entirely missing)
- "clause_location": where this clause appears (e.g. "Clause 1")
- "dpdp_section": one of the valid DPDP sections listed above (e.g. "Section 4")
- "gap_type": one of "missing", "insufficient", or "conflicting"
- "risk_level": one of "critical", "high", "medium", or "low"
- "remediation": suggested fix
- "confidence": a float between 0 and 1

Return an empty list if fully compliant. Only flag genuine gaps - do not over-report.""",
                }],
                max_retries=3,
            )
            return gaps
        except Exception as e:
            logger.warning("clause_tagging_failed", chunk_id=chunk["id"], error=str(e))
            return []

    async def _synthesize_report(self, document_id: str, chunks: list,
                                  gaps: list[ComplianceGap]) -> ComplianceReport:
        """REDUCE step: Synthesize gaps into a coherent report. Uses async litellm."""
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

        # Generate executive summary using async litellm
        summary_prompt = f"""Summarize DPDP compliance audit findings:
Total clauses analyzed: {len(chunks)}
Total gaps found: {len(gaps)}
Critical: {critical_count}, High: {high_count}
Overall risk: {overall}

Key gaps:
{json.dumps([g.model_dump() for g in gaps[:10]], indent=2)}

Write a 3-4 sentence executive summary for a legal reviewer."""

        response = await litellm.acompletion(
            model=self.OPENROUTER_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = response.choices[0].message.content

        return ComplianceReport(
            document_id=document_id,
            overall_risk=overall,
            total_clauses_analyzed=len(chunks),
            gaps=gaps,
            sections_analyzed=list(set(g.dpdp_section for g in gaps)),
            executive_summary=summary,
        )
