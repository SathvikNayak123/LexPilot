import re
from ingestion.models import ParsedDocument
from knowledge_graph.neo4j_client import Neo4jClient
import structlog

logger = structlog.get_logger()

# Lazy-loaded — only imported when first needed so the model doesn't block
# indexing if InLegalBERT is unavailable (e.g. no internet on first run).
_ner_extractor = None


def _get_ner_extractor():
    global _ner_extractor
    if _ner_extractor is None:
        try:
            from knowledge_graph.legal_ner import LegalNERExtractor
            _ner_extractor = LegalNERExtractor()
        except Exception as e:
            logger.warning("inlegalbert_unavailable", error=str(e),
                           fallback="regex-only citation extraction")
            _ner_extractor = False   # sentinel: don't retry
    return _ner_extractor if _ner_extractor is not False else None


class GraphBuilder:
    """Builds the judgment knowledge graph from parsed documents."""

    # Common Indian citation patterns
    CITATION_PATTERNS = [
        r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',          # (2017) 10 SCC 1
        r'AIR\s+\d{4}\s+SC\s+\d+',                 # AIR 2017 SC 4161
        r'\d{4}\s+SCC\s+OnLine\s+SC\s+\d+',        # 2024 SCC OnLine SC 123
        r'\[\d{4}\]\s+\d+\s+SCR\s+\d+',            # [2017] 1 SCR 123
        r'ILR\s+\d{4}\s+\w+\s+\d+',                # ILR 2020 Kar 567
    ]

    def __init__(self):
        self.neo4j = Neo4jClient()

    async def build_from_document(self, doc: ParsedDocument, judgment_metadata: dict):
        """Extract citation links from a judgment and add to graph."""
        # Add the judgment node itself
        await self.neo4j.add_judgment({
            "id": doc.document_id,
            "citation": doc.citation or judgment_metadata.get("citation", ""),
            "case_name": doc.title,
            "court": doc.court or "Unknown",
            "date": str(doc.date or "2024-01-01"),
            "bench_strength": judgment_metadata.get("bench_strength", 1),
            "subject_tags": judgment_metadata.get("subject_tags", []),
            "holding_summary": judgment_metadata.get("holding_summary", ""),
            "is_overruled": judgment_metadata.get("is_overruled", False),
            "overruled_by": judgment_metadata.get("overruled_by"),
        })

        # Bug fix #1: mark_overruled was never called.
        # If metadata declares this judgment overruled, write the OVERRULES edge
        # from the overruling case to this one.
        if judgment_metadata.get("is_overruled") and judgment_metadata.get("overruled_by"):
            overruling = await self._find_by_citation(judgment_metadata["overruled_by"])
            if overruling:
                await self.neo4j.mark_overruled(doc.document_id, overruling["id"])
                logger.info("overruled_edge_written",
                            overruled=doc.document_id, by=overruling["id"])

        # Extract relations from full text — prefer InLegalBERT typed relations,
        # fall back to plain regex CITES if the model is unavailable.
        full_text = " ".join(b.content for b in doc.blocks)
        citations_found = await self._resolve_and_store_relations(doc.document_id, full_text)

        logger.info("graph_built", doc_id=doc.document_id, citations_found=citations_found)
        return citations_found

    async def _resolve_and_store_relations(self, source_id: str, text: str,
                                            create_stubs: bool = False) -> int:
        """Extract typed relations and write edges. Returns number of edges written.

        create_stubs=True (used by the corpus-wide build script) creates lightweight
        stub nodes for cited cases not yet in Neo4j so no edges are silently dropped.
        """
        ner = _get_ner_extractor()
        count = 0

        if ner is not None:
            # InLegalBERT path: typed relations (OVERRULES, DISTINGUISHED_FROM, APPLIED, CITES)
            relations = ner.extract_relations(text)
            for rel in relations:
                target = await self._find_by_citation(rel.citation, create_stub=create_stubs)
                if target and target["id"] != source_id:
                    await self.neo4j.add_typed_citation_link(
                        source_id, target["id"],
                        rel.relation_type, rel.confidence,
                    )
                    count += 1
        else:
            # Fallback: plain regex, all edges written as CITES
            for citation in self._extract_citations(text):
                target = await self._find_by_citation(citation, create_stub=create_stubs)
                if target and target["id"] != source_id:
                    await self.neo4j.add_citation_link(source_id, target["id"])
                    count += 1

        return count

    # ------------------------------------------------------------------
    # Citation utilities
    # ------------------------------------------------------------------
    def _extract_citations(self, text: str) -> list[str]:
        """Extract unique Indian legal citations from text (regex only)."""
        citations: set[str] = set()
        for pattern in self.CITATION_PATTERNS:
            citations.update(re.findall(pattern, text))
        return list(citations)

    @staticmethod
    def _normalize_citation(citation: str) -> str:
        """Bug fix #2: normalize whitespace so minor formatting differences match."""
        return re.sub(r'\s+', ' ', citation.strip())

    async def _find_by_citation(self, citation: str, create_stub: bool = False) -> dict | None:
        """Find a judgment in the graph by citation string (with normalization).

        If create_stub=True and no node is found, a lightweight stub Judgment node
        is created so that citation edges can still be written.  The stub will be
        enriched with full data if/when the document is ingested later.
        """
        normalized = self._normalize_citation(citation)
        async with self.neo4j.driver.session() as session:
            result = await session.run(
                """MATCH (j:Judgment)
                   WHERE j.citation = $raw OR j.citation = $norm
                   RETURN j.id as id LIMIT 1""",
                raw=citation, norm=normalized,
            )
            record = await result.single()
            if record:
                return {"id": record["id"]}

            if not create_stub:
                return None

            # Create a stub node keyed by the normalised citation string
            stub_id = f"stub::{normalized}"
            await session.run(
                """MERGE (j:Judgment {id: $id})
                   ON CREATE SET j.citation = $citation,
                                 j.case_name = $citation,
                                 j.is_stub = true,
                                 j.court = 'Unknown',
                                 j.is_overruled = false
                """,
                id=stub_id, citation=normalized,
            )
            logger.debug("stub_node_created", citation=normalized[:60])
            return {"id": stub_id}
