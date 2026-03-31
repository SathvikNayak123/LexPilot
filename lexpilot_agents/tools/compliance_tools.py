from agents import function_tool
from compliance.dpdp_sections import DPDP_SECTIONS

# Lazy-initialized singletons
_scanner = None
_hybrid_pipeline = None


def _get_scanner():
    global _scanner
    if _scanner is None:
        from compliance.dpdp_scanner import DPDPScanner
        _scanner = DPDPScanner()
    return _scanner


def _get_hybrid_pipeline():
    global _hybrid_pipeline
    if _hybrid_pipeline is None:
        from retrieval.hybrid_search import HybridSearchPipeline
        _hybrid_pipeline = HybridSearchPipeline()
    return _hybrid_pipeline


@function_tool
async def scan_dpdp_compliance(document_id: str) -> str:
    """Run exhaustive DPDP Act 2023 compliance scan on a contract or privacy policy.
    Checks EVERY clause against all applicable DPDP sections.
    Returns structured compliance report with gaps and risk levels.
    This is NOT a RAG query - it processes the entire document sequentially.
    Args:
        document_id: ID of the document to scan (must be ingested first)
    """
    report = await _get_scanner().scan(document_id)
    return report.model_dump_json(indent=2)


@function_tool
def lookup_dpdp_section(section: str) -> str:
    """Look up a specific section of the DPDP Act 2023.
    Args:
        section: Section identifier, e.g., 'Section 8(1)', 'Section 16'
    """
    info = DPDP_SECTIONS.get(section)
    if not info:
        return f"Section '{section}' not found. Valid sections: {list(DPDP_SECTIONS.keys())}"
    return f"DPDP Act 2023 - {section}\nTitle: {info['title']}\nDescription: {info['description']}"


@function_tool
async def extract_clauses(document_id: str, topic: str = None) -> str:
    """Extract specific clauses from a contract document.
    Args:
        document_id: ID of the ingested contract
        topic: Optional topic filter (e.g., 'data retention', 'liability', 'termination')
    """
    results = await _get_hybrid_pipeline().search(
        query=topic or "contract clauses",
        doc_type_filter="contract",
        top_k=10,
    )
    # Filter to requested document
    doc_results = [r for r in results if r.get("document_id") == document_id]
    if not doc_results:
        doc_results = results[:5]  # Fallback to general results

    formatted = [
        f"Clause ({r.get('heading_context', 'N/A')}):\n{r.get('parent_content', '')[:400]}"
        for r in doc_results
    ]
    return "\n---\n".join(formatted) if formatted else "No clauses found."
