from agents import function_tool
from retrieval.hybrid_search import HybridSearchPipeline
from knowledge_graph.graph_retriever import GraphRAGRetriever

hybrid_pipeline = HybridSearchPipeline()
graphrag_retriever = GraphRAGRetriever()


@function_tool
async def search_legal_documents(query: str, doc_type: str = None, top_k: int = 5) -> str:
    """Search indexed legal documents using hybrid retrieval (dense + BM25 + reranking).
    Use for general legal queries, contract clause lookups, and statute searches.
    Args:
        query: Natural language search query
        doc_type: Filter by document type - "judgment", "contract", "statute", "policy"
        top_k: Number of results to return
    """
    results = await hybrid_pipeline.search(query, doc_type, top_k)
    formatted = []
    for r in results:
        formatted.append(
            f"[{r.get('doc_type', 'unknown')}] {r.get('heading_context', 'N/A')}\n"
            f"Content: {r.get('parent_content', r.get('content', ''))[:500]}\n"
            f"Source: {r.get('document_id', 'N/A')}"
        )
    return "\n---\n".join(formatted) if formatted else "No results found."


@function_tool
async def graphrag_search(query: str, top_k: int = 5) -> str:
    """Search for legal precedents using GraphRAG - combines vector similarity
    with court hierarchy and citation network analysis.
    Prioritizes: higher courts, more recent judgments, frequently cited cases.
    Flags overruled judgments.
    Args:
        query: Legal research query about precedents or case law
        top_k: Number of precedents to return
    """
    results = await graphrag_retriever.search(query, doc_type_filter="judgment", top_k=top_k)
    formatted = []
    for r in results:
        warning = f"\nWARNING: {r['warning']}" if r.get("warning") else ""
        formatted.append(
            f"Case: {r.get('heading_context', 'N/A')}\n"
            f"Court: {r.get('court', 'N/A')} | Date: {r.get('date', 'N/A')}\n"
            f"Graph Score: {r.get('graph_score', 0):.3f} | "
            f"Cited by: {r.get('cited_by_count', 0)} judgments\n"
            f"Content: {r.get('parent_content', '')[:500]}"
            f"{warning}"
        )
    return "\n---\n".join(formatted) if formatted else "No precedents found."


@function_tool
async def get_citation_chain(judgment_id: str) -> str:
    """Get the citation chain for a specific judgment - which cases it cites
    and which cases cite it. Useful for understanding precedent relationships."""
    from knowledge_graph.neo4j_client import Neo4jClient
    neo4j = Neo4jClient()
    async with neo4j.driver.session() as session:
        result = await session.run("""
            MATCH (j:Judgment {id: $jid})
            OPTIONAL MATCH (j)-[:CITES]->(cited:Judgment)
            OPTIONAL MATCH (citing:Judgment)-[:CITES]->(j)
            RETURN j.case_name as case_name,
                   collect(DISTINCT {name: cited.case_name, citation: cited.citation}) as cites,
                   collect(DISTINCT {name: citing.case_name, citation: citing.citation}) as cited_by
        """, jid=judgment_id)
        record = await result.single()
        if not record:
            return f"Judgment {judgment_id} not found in graph."
        return (
            f"Case: {record['case_name']}\n"
            f"Cites: {record['cites']}\n"
            f"Cited by: {record['cited_by']}"
        )
