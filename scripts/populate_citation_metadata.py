"""
Populate citation_index metadata — holding_summary + is_overruled
=================================================================
Syncs two critical fields that the citation verifier needs for Tier 2 and Tier 3:

  1. is_overruled / overruled_by — read from Neo4j OVERRULES edges
  2. holding_summary — generated via LLM from parent_chunks text

Without these, Tier 2 (overruled detection) and Tier 3 (mischaracterization check)
are completely non-functional — the verifier silently skips them when the columns are NULL.

Usage:
    python scripts/populate_citation_metadata.py              # Full run
    python scripts/populate_citation_metadata.py --only-overruled   # Just sync overruled flags
    python scripts/populate_citation_metadata.py --only-holdings    # Just generate holdings
    python scripts/populate_citation_metadata.py --dry-run          # Preview without writing

Requires: Postgres + Neo4j running, documents ingested.
"""

import argparse
import asyncio
import json
import os
import re
import sys

import litellm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import settings

os.environ.setdefault("OPENROUTER_API_KEY", settings.openrouter_api_key)
litellm.suppress_debug_info = True
litellm.set_verbose = False

OPENROUTER_MODEL = "openrouter/google/gemini-3-flash-preview"

HOLDING_PROMPT = """\
You are a legal research assistant. Read the judgment text below and write a ONE-SENTENCE
holding summary (max 60 words) that captures the core legal principle established.

Focus on: what was held, which right/provision was interpreted, and the practical effect.
Do NOT include case name, citation, or date — just the holding itself.

EXAMPLES:
- "The right to privacy is a fundamental right under Article 21, and any state intrusion must satisfy a proportionality test of legality, legitimate aim, necessity, and balancing."
- "Section 66A of the IT Act is unconstitutional for being vague and overbroad, having a chilling effect on free speech."
- "Parliament's amending power under Article 368 cannot destroy the basic structure of the Constitution."

JUDGMENT TEXT (first 3000 chars):
{text}

Return ONLY the holding summary sentence, nothing else."""


# Well-known overruled judgments that may not have OVERRULES edges in Neo4j.
# Format: (case_name_substring, overruled_by_citation)
KNOWN_OVERRULED = [
    ("Additional_District_Magistrate", "K.S. Puttaswamy v. Union of India (2017) 10 SCC 1"),
    ("ADM Jabalpur", "K.S. Puttaswamy v. Union of India (2017) 10 SCC 1"),
]


async def sync_overruled(dry_run: bool):
    """Read OVERRULES edges from Neo4j and update citation_index in Postgres.

    Matching strategy (Neo4j stub nodes use SCR citations as names, while
    Postgres uses file-derived case_names with SCC citations):
      1. Match by case_name (works for real nodes)
      2. Match by citation_string (works for stub nodes whose name IS a citation)

    Also applies a known-overruled-cases fallback for judgments where Neo4j
    lacks OVERRULES edges (e.g. ADM Jabalpur overruled by Puttaswamy).
    """
    from knowledge_graph.neo4j_client import Neo4jClient
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy import text

    neo4j = Neo4jClient()
    engine = create_async_engine(settings.postgres_url)

    print("\n[1] Syncing overruled status from Neo4j -> Postgres citation_index")

    # Get all OVERRULES edges from Neo4j
    async with neo4j.driver.session() as session:
        result = await session.run("""
            MATCH (new:Judgment)-[:OVERRULES]->(old:Judgment)
            RETURN old.id as overruled_id, old.case_name as overruled_name,
                   old.citation as overruled_citation,
                   new.id as overruling_id, new.case_name as overruling_name,
                   new.citation as overruling_citation
        """)
        edges = [r async for r in result]

    print(f"    Found {len(edges)} OVERRULES edges in Neo4j")

    updated = 0
    async with AsyncSession(engine) as session:
        # --- Neo4j edges ---
        for edge in edges:
            overruled_name = edge["overruled_name"]
            overruled_cit = edge["overruled_citation"]
            overruling_citation = edge["overruling_citation"] or edge["overruling_name"] or "a later judgment"

            if dry_run:
                print(f"    [DRY RUN] Would mark overruled: {overruled_name} (by {overruling_citation})")
                continue

            matched = 0

            # Strategy 1: Match by case_name
            result = await session.execute(
                text("""
                    UPDATE citation_index
                    SET is_overruled = true, overruled_by = :overruled_by
                    WHERE case_name = :case_name AND (is_overruled = false OR is_overruled IS NULL)
                    RETURNING id
                """),
                {"case_name": overruled_name, "overruled_by": overruling_citation},
            )
            matched += len(result.fetchall())

            # Strategy 2: Match by citation_string (stub nodes use citation as name)
            if overruled_cit:
                result = await session.execute(
                    text("""
                        UPDATE citation_index
                        SET is_overruled = true, overruled_by = :overruled_by
                        WHERE citation_string = :cit AND (is_overruled = false OR is_overruled IS NULL)
                        RETURNING id
                    """),
                    {"cit": overruled_cit, "overruled_by": overruling_citation},
                )
                matched += len(result.fetchall())

            if matched:
                updated += matched
                print(f"    Updated {matched} row(s): {overruled_name}")

        # --- Known-overruled fallback ---
        print("\n    Applying known-overruled-cases fallback...")
        for name_substr, overruled_by in KNOWN_OVERRULED:
            if dry_run:
                print(f"    [DRY RUN] Would mark overruled: *{name_substr}* (by {overruled_by})")
                continue

            result = await session.execute(
                text("""
                    UPDATE citation_index
                    SET is_overruled = true, overruled_by = :overruled_by
                    WHERE case_name ILIKE :pattern
                      AND (is_overruled = false OR is_overruled IS NULL)
                    RETURNING id
                """),
                {"pattern": f"%{name_substr}%", "overruled_by": overruled_by},
            )
            rows = result.fetchall()
            if rows:
                updated += len(rows)
                print(f"    Fallback updated {len(rows)} row(s): *{name_substr}*")

        if not dry_run:
            await session.commit()

    print(f"    Total rows updated: {updated}")
    await neo4j.close()


async def populate_holdings(dry_run: bool):
    """Generate holding summaries for citation_index rows that have NULL holding_summary."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy import text

    engine = create_async_engine(settings.postgres_url)

    print("\n[2] Populating holding_summary via LLM")

    # Find cases missing holding_summary
    async with AsyncSession(engine) as session:
        result = await session.execute(text("""
            SELECT ci.id, ci.case_name, ci.citation_string,
                   (SELECT string_agg(pc.content, ' ' ORDER BY pc.id)
                    FROM parent_chunks pc
                    JOIN documents d ON pc.document_id = d.id
                    WHERE d.title = ci.case_name
                    LIMIT 1) AS doc_text
            FROM citation_index ci
            WHERE ci.holding_summary IS NULL OR ci.holding_summary = ''
            ORDER BY ci.id
        """))
        rows = result.fetchall()

    print(f"    Found {len(rows)} rows missing holding_summary")

    if not rows:
        return

    generated = 0
    failed = 0

    for row_id, case_name, citation, doc_text in rows:
        if not doc_text:
            print(f"    [SKIP] No text for: {case_name} ({citation})")
            continue

        if dry_run:
            print(f"    [DRY RUN] Would generate holding for: {case_name}")
            continue

        try:
            response = await litellm.acompletion(
                model=OPENROUTER_MODEL,
                messages=[{
                    "role": "user",
                    "content": HOLDING_PROMPT.format(text=doc_text[:3000]),
                }],
                max_tokens=150,
                temperature=0,
            )
            holding = response.choices[0].message.content.strip()
            # Remove any leading quotes or "Holding:" prefix
            holding = re.sub(r'^["\']\s*', '', holding)
            holding = re.sub(r'\s*["\']$', '', holding)
            holding = re.sub(r'^(?:Holding|Summary):\s*', '', holding, flags=re.IGNORECASE)

            async with AsyncSession(engine) as session:
                await session.execute(
                    text("UPDATE citation_index SET holding_summary = :holding WHERE id = :id"),
                    {"holding": holding, "id": row_id},
                )
                await session.commit()

            generated += 1
            print(f"    [{generated}] {case_name[:50]}: {holding[:80]}...")

        except Exception as e:
            failed += 1
            print(f"    [ERROR] {case_name}: {e}")

        # Rate limiting
        if generated % 10 == 0 and generated > 0:
            await asyncio.sleep(1)

    print(f"\n    Generated: {generated}, Failed: {failed}, Skipped (no text): {len(rows) - generated - failed}")


async def main(dry_run: bool, only_overruled: bool, only_holdings: bool):
    print("=" * 60)
    print("  Citation Metadata Population")
    if dry_run:
        print("  [DRY RUN — no writes]")
    print("=" * 60)

    if not only_holdings:
        await sync_overruled(dry_run)

    if not only_overruled:
        await populate_holdings(dry_run)

    print("\n" + "=" * 60)
    print("  Done. Re-run bench_citation_accuracy.py to see improvement.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate citation_index metadata")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-overruled", action="store_true")
    parser.add_argument("--only-holdings", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run, args.only_overruled, args.only_holdings))
