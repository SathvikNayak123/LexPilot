"""
Build Knowledge Graph — Run After Ingestion
============================================
Owns the full graph lifecycle. Run this once after all documents are ingested.

Steps (all run by default):
  1. SCHEMA    — Create Neo4j indexes, constraints, and court hierarchy
  2. NODES     — Create Judgment nodes from Postgres (documents + citation_index tables)
  3. CITATIONS — Resolve citation edges across all documents (corpus-aware, no dropped edges)
  4. SEMANTIC  — Add RELATED_TO edges between judgments with high embedding similarity

Why separate from ingestion:
  Citation edges require BOTH nodes to exist. Building them per-document during ingestion
  silently drops edges whenever the cited document hasn't been ingested yet. Running
  graph build after all docs are indexed eliminates that problem entirely.

Usage:
    python scripts/build_semantic_graph.py                    # Full build (all 4 steps)
    python scripts/build_semantic_graph.py --skip-semantic    # Skip RELATED_TO edges
    python scripts/build_semantic_graph.py --only-semantic    # Only rebuild semantic edges
    python scripts/build_semantic_graph.py --only-citations   # Only rebuild citation edges
    python scripts/build_semantic_graph.py --threshold 0.65   # Looser semantic threshold
    python scripts/build_semantic_graph.py --dry-run          # Preview without writing

Requires: Postgres + Qdrant + Neo4j running.
    docker compose -f infrastructure/docker-compose.yml up -d
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Step 1: Schema & court hierarchy
# ---------------------------------------------------------------------------

async def setup_schema(neo4j):
    """Create Neo4j indexes, constraints, and Indian court hierarchy."""
    print("  Creating constraints and indexes...")
    await neo4j.setup_schema()
    print("  Setting up court hierarchy...")
    await neo4j.setup_court_hierarchy()
    print("  Schema ready.")


# ---------------------------------------------------------------------------
# Step 2: Judgment nodes from Postgres
# ---------------------------------------------------------------------------

async def _find_by_citation(neo4j, citation: str) -> dict | None:
    normalized = re.sub(r'\s+', ' ', citation.strip())
    async with neo4j.driver.session() as session:
        result = await session.run(
            "MATCH (j:Judgment) WHERE j.citation = $raw OR j.citation = $norm "
            "RETURN j.id as id LIMIT 1",
            raw=citation, norm=normalized,
        )
        record = await result.single()
        return {"id": record["id"]} if record else None


async def build_nodes(neo4j, dry_run: bool):
    """
    Read all judgment documents from Postgres and create/update Judgment nodes.

    Joins documents + citation_index on case_name = title to get the full
    picture: citation string, holding summary, overrule status.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy import text
    from config.config import settings

    engine = create_async_engine(settings.postgres_url)

    async with AsyncSession(engine) as session:
        result = await session.execute(text("""
            SELECT
                d.id,
                d.title,
                d.court,
                d.date,
                d.metadata,
                MIN(ci.citation_string)                                                    AS citation,
                MIN(ci.holding_summary)                                                    AS holding_summary,
                BOOL_OR(COALESCE(ci.is_overruled, false))                                 AS is_overruled,
                MIN(ci.overruled_by)                                                       AS overruled_by,
                ARRAY_AGG(ci.citation_string) FILTER (WHERE ci.citation_string IS NOT NULL) AS all_citations
            FROM documents d
            LEFT JOIN citation_index ci ON ci.case_name = d.title
            WHERE d.doc_type = 'judgment'
            GROUP BY d.id, d.title, d.court, d.date, d.metadata
            ORDER BY d.date
        """))
        rows = result.fetchall()

    if not rows:
        print("  No judgment documents found in Postgres. Run ingestion first.")
        return 0

    print(f"  Found {len(rows)} judgment documents.")

    if dry_run:
        for row in rows:
            print(f"    [DRY RUN] Node: {row[0][:40]}  citation={row[5]}")
        return len(rows)

    for row in rows:
        doc_id, title, court, date, metadata_raw, citation, holding, is_overruled, overruled_by, all_citations = row
        meta = metadata_raw if isinstance(metadata_raw, dict) else (json.loads(metadata_raw) if metadata_raw else {})
        primary = citation or ""
        aliases = [c for c in (all_citations or []) if c and c != primary]

        await neo4j.add_judgment({
            "id": doc_id,
            "citation": primary,
            "citation_aliases": aliases,
            "case_name": title,
            "court": court or "Unknown",
            "date": str(date) if date else "2024-01-01",
            "bench_strength": meta.get("bench_strength", 1),
            "subject_tags": meta.get("subject_tags", []),
            "holding_summary": holding or meta.get("holding_summary", ""),
            "is_overruled": is_overruled or meta.get("is_overruled", False),
            "overruled_by": overruled_by or meta.get("overruled_by"),
        })

    print(f"  Created/updated {len(rows)} Judgment nodes.")

    # Write OVERRULES edges declared via metadata
    overruled_count = 0
    for row in rows:
        doc_id, is_overruled, overruled_by_citation = row[0], row[7], row[8]
        if is_overruled and overruled_by_citation:
            overruling = await _find_by_citation(neo4j, overruled_by_citation)
            if overruling:
                await neo4j.mark_overruled(doc_id, overruling["id"])
                overruled_count += 1

    if overruled_count:
        print(f"  Wrote {overruled_count} OVERRULES edge(s) from metadata.")

    return len(rows)


# ---------------------------------------------------------------------------
# Step 3: Citation edges (corpus-aware, all nodes exist)
# ---------------------------------------------------------------------------

async def build_citation_edges(dry_run: bool):
    """
    Read full text of every judgment from Postgres and resolve citation edges.
    Because all Judgment nodes are already in Neo4j at this point, no edges are dropped.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy import text
    from knowledge_graph.graph_builder import GraphBuilder
    from config.config import settings

    engine = create_async_engine(settings.postgres_url)
    builder = GraphBuilder()

    print("  Loading parent chunks from Postgres...")
    async with AsyncSession(engine) as session:
        result = await session.execute(text("""
            SELECT document_id, string_agg(content, ' ' ORDER BY id) AS full_text
            FROM parent_chunks
            WHERE document_id IN (
                SELECT id FROM documents WHERE doc_type = 'judgment'
            )
            GROUP BY document_id
        """))
        rows = result.fetchall()

    print(f"  Resolving citations for {len(rows)} documents...")
    total_edges = 0

    for doc_id, full_text in rows:
        if dry_run:
            found = builder._extract_citations(full_text or "")
            if found:
                print(f"    [DRY RUN] {doc_id[:40]}: {len(found)} citation(s) found")
            continue
        edges = await builder._resolve_and_store_relations(doc_id, full_text or "", create_stubs=True)
        if edges:
            print(f"    {doc_id[:40]}: +{edges} edge(s)")
        total_edges += edges

    if not dry_run:
        print(f"  Total citation edges written: {total_edges}")


# ---------------------------------------------------------------------------
# Step 4: Semantic RELATED_TO edges
# ---------------------------------------------------------------------------

def collect_document_embeddings(qdrant_client, collection: str) -> dict[str, np.ndarray]:
    """Scroll Qdrant, group chunk vectors by document_id, return mean embeddings."""
    doc_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    offset = None

    print("  Scrolling Qdrant for chunk embeddings...")
    while True:
        records, offset = qdrant_client.scroll(
            collection_name=collection,
            with_vectors=["dense"],
            with_payload=["document_id", "doc_type"],
            limit=200,
            offset=offset,
        )
        for point in records:
            doc_id = point.payload.get("document_id", "")
            doc_type = point.payload.get("doc_type", "")
            if doc_id and doc_type == "judgment":
                vec = point.vector.get("dense") if isinstance(point.vector, dict) else point.vector
                if vec is not None:
                    doc_vectors[doc_id].append(np.array(vec, dtype=np.float32))
        if offset is None:
            break

    print(f"  Collected embeddings for {len(doc_vectors)} documents.")
    return {doc_id: np.mean(vecs, axis=0) for doc_id, vecs in doc_vectors.items() if vecs}


def cosine_similarity_pairs(embeddings: dict[str, np.ndarray]) -> list[tuple[str, str, float]]:
    ids = list(embeddings.keys())
    matrix = np.stack([embeddings[i] for i in ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / np.clip(norms, 1e-9, None)
    sim_matrix = normed @ normed.T
    return [
        (ids[i], ids[j], float(sim_matrix[i, j]))
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    ]


async def build_semantic_edges(neo4j, threshold: float, dry_run: bool):
    from qdrant_client import QdrantClient
    from config.config import settings

    qdrant = QdrantClient(url=settings.qdrant_url)

    async with neo4j.driver.session() as session:
        result = await session.run("MATCH (j:Judgment) RETURN j.id as id")
        judgment_ids = [r["id"] async for r in result]

    print(f"  Neo4j has {len(judgment_ids)} Judgment nodes.")
    if not judgment_ids:
        print("  No nodes found — run steps 1-3 first.")
        return

    doc_embeddings = collect_document_embeddings(qdrant, settings.qdrant_collection)
    common_ids = [i for i in judgment_ids if i in doc_embeddings]
    print(f"  Documents in both Neo4j + Qdrant: {len(common_ids)}")

    if len(common_ids) < 2:
        print("  Need at least 2 documents. Skipping.")
        return

    pairs = cosine_similarity_pairs({i: doc_embeddings[i] for i in common_ids})
    above = [(a, b, s) for a, b, s in pairs if s >= threshold]
    print(f"  Pairs above threshold {threshold}: {len(above)} / {len(pairs)}")

    if dry_run:
        print("\n  [DRY RUN] Would write these RELATED_TO edges:")
        for id_a, id_b, sim in sorted(above, key=lambda x: -x[2]):
            print(f"    {id_a[:35]} <-> {id_b[:35]}  sim={sim:.3f}")
        return

    for id_a, id_b, sim in above:
        await neo4j.add_semantic_edge(id_a, id_b, sim)
    print(f"  Written {len(above)} RELATED_TO edges.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(threshold: float, dry_run: bool,
               skip_semantic: bool, only_semantic: bool, only_citations: bool):
    from knowledge_graph.neo4j_client import Neo4jClient

    neo4j = Neo4jClient()

    print("=" * 60)
    print("  Knowledge Graph Builder")
    if dry_run:
        print("  [DRY RUN — no writes]")
    print("=" * 60)

    try:
        if only_semantic:
            print("\n[1/1] Building RELATED_TO edges...")
            await build_semantic_edges(neo4j, threshold, dry_run)

        elif only_citations:
            print("\n[1/1] Resolving citation edges...")
            await build_citation_edges(dry_run)

        else:
            print("\n[1/4] Schema & court hierarchy...")
            if not dry_run:
                await setup_schema(neo4j)
            else:
                print("  [DRY RUN] Skipping schema writes.")

            print("\n[2/4] Creating Judgment nodes...")
            await build_nodes(neo4j, dry_run)

            print("\n[3/4] Resolving citation edges...")
            await build_citation_edges(dry_run)

            if not skip_semantic:
                print("\n[4/4] Building semantic RELATED_TO edges...")
                await build_semantic_edges(neo4j, threshold, dry_run)
            else:
                print("\n[4/4] Skipped (--skip-semantic).")

    finally:
        await neo4j.close()

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the LexPilot knowledge graph")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold for RELATED_TO edges (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview actions without writing to Neo4j")
    parser.add_argument("--skip-semantic", action="store_true",
                        help="Skip RELATED_TO edge building (steps 1-3 only)")
    parser.add_argument("--only-semantic", action="store_true",
                        help="Only rebuild RELATED_TO edges (skip nodes + citations)")
    parser.add_argument("--only-citations", action="store_true",
                        help="Only rebuild citation edges (skip nodes + semantic)")
    args = parser.parse_args()

    asyncio.run(main(
        threshold=args.threshold,
        dry_run=args.dry_run,
        skip_semantic=args.skip_semantic,
        only_semantic=args.only_semantic,
        only_citations=args.only_citations,
    ))
