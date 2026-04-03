from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
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
        self.nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")
        self.similarity_floor = 0.50

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

        # Batch-fetch all citation records in a single DB round-trip
        records = await self._lookup_batch([c["citation"] for c in citations])

        results: dict[str, list] = {"verified": [], "mischaracterized": [], "fabricated": [], "overruled": []}
        clean_response = llm_response

        for cit in citations:
            record = records.get(cit["citation"])
            result = self._verify_single(cit, llm_response, record)
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

    def _verify_single(self, citation: dict, full_response: str,
                       record: dict | None) -> VerificationResult:
        """Verify a single citation through 3 tiers using a pre-fetched DB record."""
        cit_string = citation["citation"]

        # Tier 1: Does it exist?
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

    @staticmethod
    def _citation_candidates(cit: str) -> list[str]:
        """Return all known format variants of a citation for DB lookup.

        Handles four SCC formats found in practice:
          (YYYY) N REP PAGE  — modern parenthesised year (canonical)
          YYYY (N) REP PAGE  — old IndianKanoon format
          YYYY N REP PAGE    — no-parens variant (Kesavananda, etc.)
          YYYY REP (N) PAGE  — reporter-before-volume variant (ADM Jabalpur, etc.)
        Non-SCC citations are returned as-is (single-element list).
        """
        cit = cit.strip()
        candidates: list[str] = [cit]

        # SUPP volume: "YYYY SUPP (N) SCC PAGE" or "(YYYY) Supp (N) SCC PAGE"
        m = re.match(r'^(\d{4})\s+SUPP\s*\((\d+)\)\s+(SCC)\s+(.+)$', cit, re.IGNORECASE)
        if m:
            year, vol, rep, page = m.groups()
            rep = rep.upper()
            candidates.append(f"({year}) Supp ({vol}) {rep} {page}")
            candidates.append(f"({year}) {vol} {rep} {page}")
            candidates.append(f"{year} ({vol}) {rep} {page}")
            candidates.append(f"{year} Supp({vol}) {rep} {page}")
            return candidates
        m = re.match(r'^\((\d{4})\)\s+Supp\s*\((\d+)\)\s+(SCC)\s+(.+)$', cit, re.IGNORECASE)
        if m:
            year, vol, rep, page = m.groups()
            rep = rep.upper()
            candidates.append(f"{year} SUPP ({vol}) {rep} {page}")
            candidates.append(f"({year}) {vol} {rep} {page}")
            candidates.append(f"{year} ({vol}) {rep} {page}")
            return candidates

        # Modern "(YYYY) N REST" → all three legacy forms
        m = re.match(r'^\((\d{4})\)\s+(\d+)\s+(SCC|SCR|JT|SCALE)\s+(.+)$', cit)
        if m:
            year, vol, rep, page = m.groups()
            candidates.append(f"{year} ({vol}) {rep} {page}")     # old IK
            candidates.append(f"{year} {vol} {rep} {page}")        # bare/no-parens
            candidates.append(f"{year} {rep} ({vol}) {page}")      # reporter-before-vol
            return candidates

        # Old IK "YYYY (N) REP PAGE" → canonical + others
        m = re.match(r'^(\d{4})\s+\((\d+)\)\s+(SCC|SCR|JT|SCALE)\s+(.+)$', cit)
        if m:
            year, vol, rep, page = m.groups()
            candidates.append(f"({year}) {vol} {rep} {page}")
            candidates.append(f"{year} {vol} {rep} {page}")
            candidates.append(f"{year} {rep} ({vol}) {page}")
            return candidates

        # Bare "YYYY N REP PAGE" → canonical + others
        m = re.match(r'^(\d{4})\s+(\d+)\s+(SCC|SCR|JT|SCALE)\s+(.+)$', cit)
        if m:
            year, vol, rep, page = m.groups()
            candidates.append(f"({year}) {vol} {rep} {page}")
            candidates.append(f"{year} ({vol}) {rep} {page}")
            candidates.append(f"{year} {rep} ({vol}) {page}")
            return candidates

        # Reporter-before-vol "YYYY REP (N) PAGE" → canonical + others
        m = re.match(r'^(\d{4})\s+(SCC|SCR|JT|SCALE)\s+\((\d+)\)\s+(.+)$', cit)
        if m:
            year, rep, vol, page = m.groups()
            candidates.append(f"({year}) {vol} {rep} {page}")
            candidates.append(f"{year} ({vol}) {rep} {page}")
            candidates.append(f"{year} {vol} {rep} {page}")

        return candidates

    async def _lookup_batch(self, citations: list[str]) -> dict[str, dict | None]:
        """Look up multiple citations in a single DB query.

        Builds the full candidate set for all citations, fires one IN-query,
        then maps each original citation string back to its matched record (or None).
        """
        # Map each original citation to its list of format variants
        candidates_map: dict[str, list[str]] = {
            cit: self._citation_candidates(cit) for cit in citations
        }
        all_candidates = [c for variants in candidates_map.values() for c in variants]

        if not all_candidates:
            return {cit: None for cit in citations}

        async with AsyncSession(self.engine) as session:
            placeholders = ", ".join(f":c{i}" for i in range(len(all_candidates)))
            params = {f"c{i}": c for i, c in enumerate(all_candidates)}
            result = await session.execute(
                text(f"SELECT * FROM citation_index WHERE citation_string IN ({placeholders})"),
                params,
            )
            rows: dict[str, dict] = {row["citation_string"]: dict(row)
                                     for row in result.mappings().all()}

        # Resolve each original citation to the first matching variant
        return {
            cit: next((rows[v] for v in variants if v in rows), None)
            for cit, variants in candidates_map.items()
        }

    def _extract_context(self, text: str, citation: dict, window: int = 200) -> str:
        """Extract text surrounding a citation for characterization check."""
        start = max(0, citation["start"] - window)
        end = min(len(text), citation["end"] + window)
        return text[start:end]

    def _check_characterization(self, llm_context: str, actual_holding: str) -> bool:
        """Dual check: NLI contradiction detection + cosine similarity floor.

        NLI catches logical contradictions (e.g. "upheld" vs actual "struck down").
        Cosine similarity catches off-topic characterizations (e.g. describing
        a completely different aspect of the case).
        """
        # NLI check: 0=contradiction, 1=neutral, 2=entailment
        scores = self.nli_model.predict([(llm_context, actual_holding)])[0]
        contradiction, neutral, entailment = float(scores[0]), float(scores[1]), float(scores[2])

        # Cosine similarity check
        embeddings = self.embedder.encode(
            [llm_context, actual_holding], normalize_embeddings=True
        )
        sim = float(cosine_similarity(
            embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
        )[0][0])

        is_accurate = (entailment >= contradiction) and (sim >= self.similarity_floor)

        logger.debug("characterization_check",
                      nli_entail=round(entailment, 2),
                      nli_contra=round(contradiction, 2),
                      cosine_sim=round(sim, 3),
                      is_accurate=is_accurate)
        return is_accurate
