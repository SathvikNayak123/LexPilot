import asyncio
import json
import os
import re
from pathlib import Path

import litellm
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

from config.config import settings
from ingestion.pdf_parser import PDFParser
from ingestion.models import ParsedDocument

logger = structlog.get_logger()

os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
litellm.suppress_debug_info = True
litellm.set_verbose = False

_CITATION_PROMPT = """\
Extract the citation strings that identify THIS judgment from the text below.
Return ONLY valid JSON:
{{"primary": "<most authoritative citation>", "aliases": ["<other major-reporter citations>"]}}

EXAMPLES (few-shot):

Example 1 — Modern SCC preferred as primary:
Text: "CITATION: (2017) 10 SCC 1  AIR 2017 SC 4161  [2017] 8 SCR 1"
Output: {{"primary": "(2017) 10 SCC 1", "aliases": ["AIR 2017 SC 4161", "[2017] 8 SCR 1"]}}

Example 2 — Old IndianKanoon year-first formats, SCC still primary:
Text: "1976 AIR 1207  1976 SCR 172  1976 SCC (2) 521"
Output: {{"primary": "1976 SCC (2) 521", "aliases": ["1976 AIR 1207", "1976 SCR 172"]}}

Example 3 — AIR as primary when no SCC:
Text: "AIR 1987 SC 1086  [1987] 1 SCR 819"
Output: {{"primary": "AIR 1987 SC 1086", "aliases": ["[1987] 1 SCR 819"]}}

Example 4 — CITATOR INFO block must be excluded entirely:
Text: "(1997) 6 SCC 241  CITATOR INFO: RF (2001) 3 SCC 12  R (2010) 5 SCC 600"
Output: {{"primary": "(1997) 6 SCC 241", "aliases": []}}

Example 5 — Single citation, no aliases:
Text: "(2018) 10 SCC 1"
Output: {{"primary": "(2018) 10 SCC 1", "aliases": []}}

Rules:
- Only include citations that identify THIS document, not cases it references
- Citations appear in a CITATION block, REPORTABLE header, or cause-title near the top
- CITATOR INFO / RF / R / E entries list OTHER cases — exclude them entirely
- Prefer SCC print as primary; then SCR, then AIR
- Aliases: include ONLY major reporters — SCC, SCC OnLine, SCR, AIR, JT, SCALE, MANU, ILR
- Skip obscure/regional reporters (CLC, LRI, UJ, SRJ, GUJ LR, RECCIVR, ICC, etc.)
- Old IndianKanoon formats: "1976 AIR 1207", "1976 SCR 172", "1976 SCC (2) 521"
- Modern formats: "(2017) 10 SCC 1", "AIR 2017 SC 4161", "[2017] 1 SCR 123"
- If no citations found return {{"primary": null, "aliases": []}}

TEXT:
{text}"""


class DocumentProcessor:
    """Orchestrates document ingestion: parse PDF -> extract citation -> store metadata."""

    OPENROUTER_MODEL = "openrouter/google/gemini-3-flash-preview"

    def __init__(self):
        self.parser = PDFParser()
        self.engine = create_async_engine(settings.postgres_url)

    async def ingest(self, pdf_path: str, doc_type: str = "judgment",
                     source: str = None, court: str = None,
                     citation: str = None) -> ParsedDocument:
        """Parse a PDF, extract citations via LLM, and store its metadata."""
        # PDFParser.parse() is sync/CPU-bound (fitz + pdfplumber) — run in thread pool
        # so it doesn't block the event loop during parallel ingestion.
        doc = await asyncio.to_thread(self.parser.parse, pdf_path, doc_type, source, court)

        # Citation extraction: async LLM call on page-1 text only.
        # Done here (not inside parse()) so the thread pool stays purely CPU-bound.
        if citation:
            # Caller supplied a citation override — use it directly with no aliases.
            doc.citation = citation
        else:
            first_page_text = " ".join(
                b.content for b in doc.blocks if b.page_number == 1
            )
            doc.citation, doc.citation_aliases = await self._llm_extract_citations(first_page_text)

        async with AsyncSession(self.engine) as session:
            await session.execute(
                text("""
                    INSERT INTO documents (id, title, doc_type, source, court, date, metadata, parse_duration_ms)
                    VALUES (:id, :title, :dt, :src, :court, :date, :meta, :dur)
                    ON CONFLICT (id) DO UPDATE SET
                        title = :title, metadata = :meta, parse_duration_ms = :dur
                """),
                {
                    "id": doc.document_id, "title": doc.title,
                    "dt": doc.doc_type, "src": doc.source,
                    "court": doc.court, "date": str(doc.date) if doc.date else None,
                    "meta": json.dumps(doc.metadata),
                    "dur": doc.parse_duration_ms,
                },
            )
            await session.commit()

        logger.info("document_ingested", doc_id=doc.document_id,
                     title=doc.title, citation=doc.citation, blocks=len(doc.blocks))
        return doc

    async def _llm_extract_citations(self, text: str) -> tuple[str | None, list[str]]:
        """Extract primary citation and aliases from first-page text using an LLM.

        Falls back to regex (_extract_citations_block) if the LLM call fails or
        returns unparseable output.
        """
        if not text.strip():
            return None, []

        try:
            response = await litellm.acompletion(
                model=self.OPENROUTER_MODEL,
                messages=[{
                    "role": "user",
                    "content": _CITATION_PROMPT.format(text=text[:3000]),
                }],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()

            # Try direct parse first; if the model returns surrounding prose or
            # a trailing comma, extract the first {...} block and retry.
            data = None
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group())
                    except json.JSONDecodeError:
                        pass

            if data is None:
                raise ValueError(f"Unparseable LLM response: {raw[:120]}")

            primary = data.get("primary") or None
            aliases = [a for a in (data.get("aliases") or []) if a and a != primary]
            logger.debug("llm_citation_extracted", primary=primary, aliases=aliases)
            return primary, aliases

        except Exception as e:
            logger.warning("llm_citation_extraction_failed", error=str(e),
                           fallback="regex")
            return self.parser._extract_citations_block(text)

    async def ingest_directory(self, directory: str, doc_type: str = "judgment",
                                source: str = None, court: str = None) -> list[ParsedDocument]:
        """Ingest all PDFs in a directory."""
        docs = []
        for pdf_path in sorted(Path(directory).glob("*.pdf")):
            try:
                doc = await self.ingest(str(pdf_path), doc_type, source, court)
                docs.append(doc)
            except Exception as e:
                logger.error("ingestion_failed", path=str(pdf_path), error=str(e))
        return docs
