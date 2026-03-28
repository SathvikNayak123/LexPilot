import re
from ingestion.models import ParsedDocument
from knowledge_graph.neo4j_client import Neo4jClient
import structlog

logger = structlog.get_logger()


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
        # Add the judgment itself
        await self.neo4j.add_judgment({
            "id": doc.document_id,
            "citation": judgment_metadata.get("citation", ""),
            "case_name": doc.title,
            "court": doc.court or "Unknown",
            "date": str(doc.date or "2024-01-01"),
            "bench_strength": judgment_metadata.get("bench_strength", 1),
            "subject_tags": judgment_metadata.get("subject_tags", []),
            "holding_summary": judgment_metadata.get("holding_summary", ""),
            "is_overruled": False,
            "overruled_by": None,
        })

        # Extract citations from text
        full_text = " ".join(b.content for b in doc.blocks)
        cited_judgments = self._extract_citations(full_text)

        for citation in cited_judgments:
            # Try to find cited judgment in graph
            existing = await self._find_by_citation(citation)
            if existing:
                await self.neo4j.add_citation_link(doc.document_id, existing["id"])
                logger.info("citation_linked", from_=doc.document_id, to=existing["id"])

        logger.info("graph_built", doc_id=doc.document_id, citations_found=len(cited_judgments))

    def _extract_citations(self, text: str) -> list[str]:
        """Extract Indian legal citations from text."""
        citations = set()
        for pattern in self.CITATION_PATTERNS:
            matches = re.findall(pattern, text)
            citations.update(matches)
        return list(citations)

    async def _find_by_citation(self, citation: str) -> dict | None:
        """Find a judgment in the graph by citation string."""
        async with self.neo4j.driver.session() as session:
            result = await session.run(
                "MATCH (j:Judgment {citation: $cit}) RETURN j.id as id LIMIT 1",
                cit=citation,
            )
            record = await result.single()
            return {"id": record["id"]} if record else None
