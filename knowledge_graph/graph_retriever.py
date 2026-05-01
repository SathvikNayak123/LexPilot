import math
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
        """Enrich vector search results with graph metadata.

        Batches all Neo4j lookups into 2 queries (one for node metadata,
        one for citation chains) instead of 2×N individual queries.
        """
        doc_ids = [r.get("document_id", "") for r in results]

        async with self.neo4j.driver.session() as session:
            # --- Batch 1: node metadata for all doc_ids at once ---
            meta_result = await session.run("""
                UNWIND $doc_ids AS doc_id
                MATCH (j:Judgment {id: doc_id})
                OPTIONAL MATCH (j)-[:DECIDED_BY]->(c:Court)
                OPTIONAL MATCH (j)<-[:CITES]-(citing:Judgment)
                OPTIONAL MATCH (j)-[:CITES]->(cited:Judgment)
                RETURN doc_id,
                       j.citation      AS citation,
                       c.level         AS court_level,
                       count(DISTINCT citing) AS cited_by_count,
                       count(DISTINCT cited)  AS cites_count,
                       j.is_overruled  AS is_overruled,
                       j.overruled_by  AS overruled_by,
                       j.date          AS judgment_date
            """, doc_ids=doc_ids)

            meta_map: dict[str, dict] = {}
            async for record in meta_result:
                meta_map[record["doc_id"]] = dict(record)

            # --- Batch 2: citation chains for all doc_ids at once ---
            # Do NOT traverse DISTINGUISHED_FROM — distinguished cases are
            # explicitly not analogous.
            # Bug fix #5: COALESCE so null is_overruled doesn't drop nodes.
            chain_result = await session.run("""
                UNWIND $doc_ids AS doc_id
                MATCH (j:Judgment {id: doc_id})-[:CITES|APPLIED|RELATED_TO*1..2]-(related:Judgment)
                WHERE COALESCE(related.is_overruled, false) = false
                OPTIONAL MATCH (related)-[:DECIDED_BY]->(c:Court)
                RETURN DISTINCT doc_id,
                       related.id           AS id,
                       related.case_name    AS case_name,
                       related.citation     AS citation,
                       related.court        AS court,
                       related.date         AS date,
                       c.level              AS court_level,
                       related.holding_summary AS holding
                ORDER BY c.level ASC, related.date DESC
            """, doc_ids=doc_ids)

            chain_map: dict[str, list[dict]] = {did: [] for did in doc_ids}
            async for record in chain_result:
                did = record["doc_id"]
                if len(chain_map[did]) < 10:
                    chain_map[did].append({
                        "id": record["id"],
                        "case_name": record["case_name"],
                        "citation": record["citation"],
                        "court": record["court"],
                        "court_level": record["court_level"],
                        "holding": record["holding"],
                    })

        # Merge enriched data back into results
        for r in results:
            doc_id = r.get("document_id", "")
            meta = meta_map.get(doc_id)
            if meta:
                # Do NOT overwrite r["citation"] — the Qdrant payload stores the
                # LLM-extracted primary citation set at indexing time, which is the
                # authoritative value used for benchmark matching and deduplication.
                # Neo4j enrichment only supplies graph-specific metadata.
                if not r.get("citation") and meta["citation"]:
                    r["citation"] = meta["citation"]
                r["court_level"] = meta["court_level"] or 3
                r["cited_by_count"] = meta["cited_by_count"]
                r["cites_count"] = meta["cites_count"]
                r["is_overruled"] = meta["is_overruled"] or False
                r["overruled_by"] = meta["overruled_by"]
                r["judgment_date"] = str(meta["judgment_date"]) if meta["judgment_date"] else None
            else:
                # Default to Supreme Court (level 1) since all ingested docs are SC judgments
                r["court_level"] = 1
                r["cited_by_count"] = 0
                r["is_overruled"] = False
                logger.debug("graph_node_not_found", document_id=doc_id,
                             note="run build_semantic_graph.py to populate Neo4j")
            r["related_judgments"] = chain_map.get(doc_id, [])

        return results

    def _graph_rerank(self, results: list[dict]) -> list[dict]:
        """Re-rank using semantic relevance as primary signal with multiplicative graph boosts.

        Design rationale:
        - Semantic score (cross-encoder rerank + RRF) is the PRIMARY ranking signal.
        - Graph signals act as small MULTIPLICATIVE boosts, not additive competitors.
          This preserves the semantic ordering while giving an edge to authoritative,
          highly-cited, non-overruled judgments.
        - Recency is intentionally excluded: in constitutional law, older landmark
          precedents (Kesavananda 1973, Maneka Gandhi 1978) are often MORE authoritative
          than recent cases.  An exponential decay penalty actively hurts retrieval.
        """
        if not results:
            return results

        # Min-max normalize rerank_score to [0, 1] within the candidate set
        raw_reranks = [r.get("rerank_score", 0.0) for r in results]
        min_r, max_r = min(raw_reranks), max(raw_reranks)
        rerank_range = max_r - min_r or 1.0

        raw_rrfs = [r.get("rrf_score", 0.0) for r in results]
        min_rrf, max_rrf = min(raw_rrfs), max(raw_rrfs)
        rrf_range = max_rrf - min_rrf or 1.0

        for r in results:
            rerank = (r.get("rerank_score", 0.0) - min_r) / rerank_range
            rrf = (r.get("rrf_score", 0.0) - min_rrf) / rrf_range

            # Base semantic score: blend of reranker and RRF
            semantic = rerank * 0.75 + rrf * 0.25

            # Court authority boost: SC (1.0) > HC (0.95) > District (0.90)
            court_level = r.get("court_level", 3)
            authority_boost = {1: 1.0, 2: 0.95, 3: 0.90, 4: 0.85}.get(court_level, 0.90)

            # Citation importance boost: highly-cited cases get a small edge
            # log1p(0)=0 → 1.0x, log1p(10)=2.4 → 1.05x, log1p(50)=3.9 → 1.08x
            cited_by = r.get("cited_by_count", 0)
            citation_boost = 1.0 + math.log1p(cited_by) * 0.02

            # Overruled penalty: critical for legal correctness
            overruled_penalty = 0.1 if r.get("is_overruled") else 1.0

            r["graph_score"] = semantic * authority_boost * citation_boost * overruled_penalty

        return sorted(results, key=lambda r: r["graph_score"], reverse=True)

