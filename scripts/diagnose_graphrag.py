"""
GraphRAG Diagnostic — Check why Neo4j graph enrichment is producing no lift.
Run: python scripts/diagnose_graphrag.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from qdrant_client import QdrantClient
    from knowledge_graph.neo4j_client import Neo4jClient
    from config.config import settings

    print("=" * 60)
    print("  GraphRAG Diagnostic")
    print("=" * 60)

    neo4j = Neo4jClient()

    # ── 1. Count Neo4j Judgment nodes ───────────────────────────
    print("\n[1] Neo4j Judgment nodes")
    try:
        async with neo4j.driver.session() as session:
            r = await session.run("MATCH (j:Judgment) RETURN count(j) as n")
            rec = await r.single()
            count = rec["n"] if rec else 0
        print(f"    Count: {count}")

        if count > 0:
            async with neo4j.driver.session() as session:
                r = await session.run("MATCH (j:Judgment) RETURN j.id as id LIMIT 5")
                records = [x async for x in r]
            print(f"    Sample IDs:")
            for rec in records:
                print(f"      {rec['id']}")
        else:
            print("    *** NO NODES FOUND — build_semantic_graph.py step 2 did not write nodes ***")
            print("    Check: does 'documents' table have rows with doc_type='judgment'?")
    except Exception as e:
        print(f"    ERROR connecting to Neo4j: {e}")
        print(f"    neo4j_uri = {settings.neo4j_uri}")
        await neo4j.close()
        return

    # ── 2. Sample Qdrant document_ids ───────────────────────────
    print("\n[2] Qdrant document_ids (sample)")
    try:
        qdrant = QdrantClient(url=settings.qdrant_url)
        points, _ = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=["document_id", "doc_type"],
            with_vectors=False,
            limit=5,
        )
        judgment_ids = []
        for p in points:
            doc_id = p.payload.get("document_id", "")
            doc_type = p.payload.get("doc_type", "")
            print(f"    doc_type={doc_type}  document_id={doc_id}")
            if doc_type == "judgment":
                judgment_ids.append(doc_id)
    except Exception as e:
        print(f"    ERROR connecting to Qdrant: {e}")
        await neo4j.close()
        return

    # ── 3. Cross-check: do Qdrant IDs exist in Neo4j? ───────────
    print("\n[3] Cross-check: Qdrant document_ids vs Neo4j nodes")
    if judgment_ids and count > 0:
        async with neo4j.driver.session() as session:
            for doc_id in judgment_ids[:5]:
                r = await session.run(
                    "MATCH (j:Judgment {id: $id}) RETURN j.id as id", id=doc_id
                )
                rec = await r.single()
                found = "FOUND ✓" if rec else "NOT FOUND ✗"
                print(f"    {doc_id}  →  {found}")
    elif count == 0:
        print("    (Skipped — Neo4j has no nodes)")
    else:
        print("    (No judgment-type points found in Qdrant sample)")

    # ── 4. Check CITES/RELATED_TO edges ─────────────────────────
    if count > 0:
        print("\n[4] Graph edges")
        async with neo4j.driver.session() as session:
            for rel in ["CITES", "APPLIED", "RELATED_TO", "OVERRULES"]:
                r = await session.run(f"MATCH ()-[:{rel}]->() RETURN count(*) as n")
                rec = await r.single()
                print(f"    {rel}: {rec['n'] if rec else 0}")

    await neo4j.close()
    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
