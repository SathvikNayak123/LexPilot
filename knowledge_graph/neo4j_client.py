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
        """Add a CITES edge with temporal weight."""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (a:Judgment {id: $citing}), (b:Judgment {id: $cited})
                MERGE (a)-[r:CITES]->(b)
                SET r.weight = 1.0
            """, citing=citing_id, cited=cited_id)

    async def mark_overruled(self, overruled_id: str, overruling_id: str):
        """Mark a judgment as overruled."""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (old:Judgment {id: $old_id}), (new:Judgment {id: $new_id})
                SET old.is_overruled = true, old.overruled_by = $new_id
                MERGE (new)-[:OVERRULES]->(old)
            """, old_id=overruled_id, new_id=overruling_id)
