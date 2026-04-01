import math
from datetime import date
from knowledge_graph.neo4j_client import Neo4jClient
from retrieval.hybrid_search import HybridSearchPipeline
import structlog

logger = structlog.get_logger()


class GraphRAGRetriever:
    """Combines vector retrieval with knowledge graph traversal."""

    # Court authority weights
    COURT_WEIGHTS = {
        1: 1.0,    # Supreme Court
        2: 0.7,    # High Court
        3: 0.4,    # District Court
        4: 0.3,    # Tribunal
    }

    def __init__(self):
        self.hybrid = HybridSearchPipeline()
        self.neo4j = Neo4jClient()

    async def search(self, query: str, doc_type_filter: str = None,
                     top_k: int = 5) -> list[dict]:
        """Full GraphRAG search: vector retrieval + graph re-ranking."""

        # Step 1: Hybrid vector search (dense + sparse + RRF + rerank)
        vector_results = await self.hybrid.search(query, doc_type_filter, top_k=20)

        if not vector_results:
            return []

        # Step 2: Enrich with graph data
        enriched = await self._enrich_with_graph(vector_results)

        # Step 3: Graph-aware re-ranking
        reranked = self._graph_rerank(enriched)

        # Step 4: Flag overruled judgments
        for r in reranked:
            if r.get("is_overruled"):
                r["warning"] = f"This judgment has been overruled by {r.get('overruled_by', 'a later judgment')}"

        return reranked[:top_k]

    async def _enrich_with_graph(self, results: list[dict]) -> list[dict]:
        """Enrich vector search results with graph metadata."""
        async with self.neo4j.driver.session() as session:
            for r in results:
                doc_id = r.get("document_id", "")
                # Fetch judgment metadata from graph
                result = await session.run("""
                    MATCH (j:Judgment {id: $doc_id})
                    OPTIONAL MATCH (j)-[:DECIDED_BY]->(c:Court)
                    OPTIONAL MATCH (j)<-[:CITES]-(citing:Judgment)
                    OPTIONAL MATCH (j)-[:CITES]->(cited:Judgment)
                    RETURN j, c.level as court_level,
                           j.citation as citation,
                           count(DISTINCT citing) as cited_by_count,
                           count(DISTINCT cited) as cites_count,
                           j.is_overruled as is_overruled,
                           j.overruled_by as overruled_by,
                           j.date as judgment_date
                """, doc_id=doc_id)

                record = await result.single()
                if record:
                    if record["citation"]:
                        r["citation"] = record["citation"]
                    r["court_level"] = record["court_level"] or 3
                    r["cited_by_count"] = record["cited_by_count"]
                    r["cites_count"] = record["cites_count"]
                    r["is_overruled"] = record["is_overruled"] or False
                    r["overruled_by"] = record["overruled_by"]
                    r["judgment_date"] = str(record["judgment_date"]) if record["judgment_date"] else None
                else:
                    # Default to Supreme Court (level 1) since all ingested docs are SC judgments
                    r["court_level"] = 1
                    r["cited_by_count"] = 0
                    r["is_overruled"] = False

            # Find related judgments via citation chains
            for r in results:
                related = await self._get_citation_chain(session, r.get("document_id", ""))
                r["related_judgments"] = related

        return results

    async def _get_citation_chain(self, session, doc_id: str, depth: int = 2) -> list[dict]:
        """Traverse citation graph to find related judgments.

        Traverses CITES, APPLIED, and RELATED_TO edges (but not DISTINGUISHED_FROM —
        a distinguished case is explicitly not analogous).
        Bug fix #5: uses COALESCE so null is_overruled values don't silently drop nodes.
        """
        result = await session.run("""
            MATCH (j:Judgment {id: $doc_id})-[:CITES|APPLIED|RELATED_TO*1..2]-(related:Judgment)
            WHERE COALESCE(related.is_overruled, false) = false
            OPTIONAL MATCH (related)-[:DECIDED_BY]->(c:Court)
            RETURN DISTINCT related.id as id, related.case_name as case_name,
                   related.citation as citation, related.court as court,
                   related.date as date, c.level as court_level,
                   related.holding_summary as holding
            ORDER BY c.level ASC, related.date DESC
            LIMIT 10
        """, doc_id=doc_id)

        records = [r async for r in result]
        return [
            {
                "id": rec["id"], "case_name": rec["case_name"],
                "citation": rec["citation"], "court": rec["court"],
                "court_level": rec["court_level"],
                "holding": rec["holding"],
            }
            for rec in records
        ]

    def _graph_rerank(self, results: list[dict]) -> list[dict]:
        """Re-rank using: rrf_score * court_authority * recency * citation_importance."""
        for r in results:
            rrf = r.get("rrf_score", 0.5)
            rerank = r.get("rerank_score", 0.5)

            # Court authority weight
            court_level = r.get("court_level", 3)
            authority = self.COURT_WEIGHTS.get(court_level, 0.3)

            # Recency weight (exponential decay, half-life = 10 years)
            recency = self._recency_weight(r.get("judgment_date"))

            # Citation importance (log of citation count)
            cited_by = r.get("cited_by_count", 0)
            citation_importance = 1.0 + math.log1p(cited_by) * 0.2

            # Overruled penalty
            overruled_penalty = 0.1 if r.get("is_overruled") else 1.0

            r["graph_score"] = (
                rerank * 0.4 +
                authority * 0.25 +
                recency * 0.15 +
                citation_importance * 0.1 +
                rrf * 0.1
            ) * overruled_penalty

        return sorted(results, key=lambda r: r["graph_score"], reverse=True)

    def _recency_weight(self, date_str: str | None) -> float:
        """Exponential decay weight. Half-life = 10 years."""
        if not date_str:
            return 0.5
        try:
            d = date.fromisoformat(date_str)
            years_ago = (date.today() - d).days / 365.25
            return math.exp(-0.069 * years_ago)  # ln(2)/10 ~ 0.069
        except ValueError:
            return 0.5
