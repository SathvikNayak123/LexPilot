from neo4j import AsyncGraphDatabase
from datetime import date, datetime
import math
import structlog

from config.config import settings

logger = structlog.get_logger()


class Neo4jClient:
    """Neo4j connection and graph operations."""

    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self):
        await self.driver.close()

    async def setup_schema(self):
        """Create indexes and constraints."""
        async with self.driver.session() as session:
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (j:Judgment) REQUIRE j.id IS UNIQUE")
            await session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Court) REQUIRE c.name IS UNIQUE")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (j:Judgment) ON (j.citation)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (j:Judgment) ON (j.is_overruled)")

    async def setup_court_hierarchy(self):
        """Initialize the Indian court hierarchy."""
        courts = [
            ("Supreme Court of India", 1),
            ("Delhi High Court", 2), ("Bombay High Court", 2),
            ("Madras High Court", 2), ("Calcutta High Court", 2),
            ("Karnataka High Court", 2), ("Allahabad High Court", 2),
            ("Kerala High Court", 2), ("Gujarat High Court", 2),
            ("Punjab and Haryana High Court", 2),
            ("Telangana High Court", 2),
        ]

        async with self.driver.session() as session:
            for name, level in courts:
                await session.run(
                    "MERGE (c:Court {name: $name}) SET c.level = $level",
                    name=name, level=level,
                )

            # Supreme Court is superior to all High Courts
            await session.run("""
                MATCH (sc:Court {level: 1}), (hc:Court {level: 2})
                MERGE (sc)-[:SUPERIOR_TO]->(hc)
            """)

    async def add_judgment(self, judgment: dict):
        """Add a judgment node to the graph."""
        async with self.driver.session() as session:
            await session.run("""
                MERGE (j:Judgment {id: $id})
                SET j.citation = $citation,
                    j.case_name = $case_name,
                    j.court = $court,
                    j.date = date($date),
                    j.bench_strength = $bench_strength,
                    j.subject_tags = $subject_tags,
                    j.holding_summary = $holding_summary,
                    j.is_overruled = $is_overruled,
                    j.overruled_by = $overruled_by
                WITH j
                MATCH (c:Court {name: $court})
                MERGE (j)-[:DECIDED_BY]->(c)
            """, **judgment)

    async def add_citation_link(self, citing_id: str, cited_id: str):
        """Add a plain CITES edge (fallback when InLegalBERT is unavailable)."""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (a:Judgment {id: $citing}), (b:Judgment {id: $cited})
                MERGE (a)-[r:CITES]->(b)
                SET r.weight = 1.0, r.confidence = 0.5
            """, citing=citing_id, cited=cited_id)

    # Whitelisted relation types — never interpolate user input into Cypher
    _ALLOWED_RELATIONS = frozenset({"CITES", "OVERRULES", "DISTINGUISHED_FROM", "APPLIED"})

    async def add_typed_citation_link(self, citing_id: str, cited_id: str,
                                      relation_type: str, confidence: float = 1.0):
        """Add a typed edge (CITES / OVERRULES / DISTINGUISHED_FROM / APPLIED).

        relation_type is whitelisted; unknown values fall back to CITES.
        If the relation is OVERRULES, the cited node is also flagged is_overruled=true.
        """
        if relation_type not in self._ALLOWED_RELATIONS:
            relation_type = "CITES"

        query = f"""
            MATCH (a:Judgment {{id: $citing}}), (b:Judgment {{id: $cited}})
            MERGE (a)-[r:{relation_type}]->(b)
            SET r.weight = 1.0, r.confidence = $confidence
        """
        async with self.driver.session() as session:
            await session.run(query, citing=citing_id, cited=cited_id, confidence=confidence)

            if relation_type == "OVERRULES":
                await session.run("""
                    MATCH (old:Judgment {id: $cited})
                    SET old.is_overruled = true, old.overruled_by = $citing
                """, cited=cited_id, citing=citing_id)

    async def add_semantic_edge(self, id_a: str, id_b: str, similarity: float):
        """Add a RELATED_TO edge based on embedding cosine similarity."""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (a:Judgment {id: $id_a}), (b:Judgment {id: $id_b})
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.similarity = $sim
            """, id_a=id_a, id_b=id_b, sim=similarity)

    async def mark_overruled(self, overruled_id: str, overruling_id: str):
        """Mark a judgment as overruled and write the OVERRULES edge."""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (old:Judgment {id: $old_id}), (new:Judgment {id: $new_id})
                SET old.is_overruled = true, old.overruled_by = $new_id
                MERGE (new)-[:OVERRULES]->(old)
            """, old_id=overruled_id, new_id=overruling_id)
