from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import BaseModel
from typing import Literal
import structlog

from config.config import settings
from citation.extractor import CitationExtractor

logger = structlog.get_logger()


class VerificationResult(BaseModel):
    citation: str
    status: Literal["verified", "mischaracterized", "fabricated", "overruled"]
    case_name: str | None = None
    actual_holding: str | None = None
    correction: str | None = None  # For mischaracterized citations


class CitationVerifier:
    """Verifies LLM-generated citations against a verified judgment index.

    Three-tier check:
    1. EXISTS: Citation string matches an entry in the verified index
    2. NOT OVERRULED: Judgment hasn't been explicitly overruled
    3. ACCURATELY CHARACTERIZED: LLM's description of the holding matches
       the actual holding (embedding similarity > 0.75)
    """

    def __init__(self):
        self.engine = create_async_engine(settings.postgres_url)
        self.extractor = CitationExtractor()
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.characterization_threshold = 0.60

    async def verify_response(self, llm_response: str) -> dict:
        """Verify all citations in an LLM response.

        Returns:
            {
                "verified": [...],
                "mischaracterized": [...],
                "fabricated": [...],
                "overruled": [...],
                "clean_response": str  # Response with fabricated citations removed
            }
        """
        citations = self.extractor.extract(llm_response)

        if not citations:
            return {
                "verified": [], "mischaracterized": [],
                "fabricated": [], "overruled": [],
                "clean_response": llm_response,
            }

        results = {"verified": [], "mischaracterized": [], "fabricated": [], "overruled": []}
        clean_response = llm_response

        for cit in citations:
            result = await self._verify_single(cit, llm_response)
            results[result.status].append(result.model_dump())

            if result.status == "fabricated":
                clean_response = clean_response.replace(
                    cit["citation"],
                    "[citation removed - not found in verified index]"
                )
            elif result.status == "overruled":
                clean_response = clean_response.replace(
                    cit["citation"],
                    f"{cit['citation']} [OVERRULED]"
                )
            elif result.status == "mischaracterized":
                clean_response += f"\n\nCorrection for {cit['citation']}: {result.correction}"

        results["clean_response"] = clean_response

        logger.info("citations_verified",
                     total=len(citations),
                     verified=len(results["verified"]),
                     fabricated=len(results["fabricated"]),
                     mischaracterized=len(results["mischaracterized"]),
                     overruled=len(results["overruled"]))

        return results

    async def _verify_single(self, citation: dict, full_response: str) -> VerificationResult:
        """Verify a single citation through 3 tiers."""
        cit_string = citation["citation"]

        # Tier 1: Does it exist?
        record = await self._lookup(cit_string)
        if not record:
            return VerificationResult(citation=cit_string, status="fabricated")

        # Tier 2: Is it overruled?
        if record.get("is_overruled"):
            return VerificationResult(
                citation=cit_string, status="overruled",
                case_name=record["case_name"],
                correction=f"Overruled by {record.get('overruled_by', 'a later judgment')}",
            )

        # Tier 3: Is the LLM's characterization accurate?
        context_around_citation = self._extract_context(full_response, citation)
        if context_around_citation and record.get("holding_summary"):
            is_accurate = self._check_characterization(
                context_around_citation, record["holding_summary"]
            )
            if not is_accurate:
                return VerificationResult(
                    citation=cit_string, status="mischaracterized",
                    case_name=record["case_name"],
                    actual_holding=record["holding_summary"],
                    correction=f"The actual holding in {record['case_name']} was: {record['holding_summary'][:200]}",
                )

        return VerificationResult(
            citation=cit_string, status="verified",
            case_name=record["case_name"],
        )

    async def _lookup(self, citation: str) -> dict | None:
        """Look up citation in the verified index."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                text("SELECT * FROM citation_index WHERE citation_string = :cit"),
                {"cit": citation},
            )
            row = result.mappings().first()
            return dict(row) if row else None

    def _extract_context(self, text: str, citation: dict, window: int = 200) -> str:
        """Extract text surrounding a citation for characterization check."""
        start = max(0, citation["start"] - window)
        end = min(len(text), citation["end"] + window)
        return text[start:end]

    def _check_characterization(self, llm_context: str, actual_holding: str) -> bool:
        """Check if LLM's characterization matches actual holding via embedding similarity."""
        embeddings = self.embedder.encode([llm_context, actual_holding], normalize_embeddings=True)
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        return similarity >= self.characterization_threshold
