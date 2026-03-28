from pydantic import BaseModel
from typing import Literal
import structlog

logger = structlog.get_logger()


class ConfidenceAssessment(BaseModel):
    tier: Literal["green", "yellow", "red"]
    score: float  # 0-1
    reasoning: str
    sources_count: int
    conflicting_sources: bool


class ConfidenceCalibrator:
    """Calibrates and surfaces uncertainty in legal AI responses.

    Tiers:
    - GREEN: High confidence - multiple supporting authorities, consistent findings
    - YELLOW: Moderate - single source, or minor ambiguity in law
    - RED: Low - conflicting authorities, no direct precedent, or novel legal question

    For RED tier: explicitly state what the system is uncertain about
    and recommend consulting a qualified advocate.
    """

    def assess(
        self, query: str, results: list[dict], response: str
    ) -> ConfidenceAssessment:
        """Assess confidence of a response based on supporting evidence."""
        sources = len(results)
        courts = set(r.get("court", "") for r in results)
        has_supreme_court = any("Supreme Court" in c for c in courts)
        has_overruled = any(r.get("is_overruled") for r in results)
        has_conflicting = self._detect_conflicting(results)

        # Scoring
        score = 0.5  # Base

        # Source count
        if sources >= 3:
            score += 0.2
        elif sources >= 1:
            score += 0.1
        else:
            score -= 0.3

        # Court authority
        if has_supreme_court:
            score += 0.2

        # Negative signals
        if has_conflicting:
            score -= 0.3
        if has_overruled:
            score -= 0.2

        # Clamp
        score = max(0.0, min(1.0, score))

        # Determine tier
        if score >= 0.7:
            tier = "green"
            reasoning = f"High confidence: {sources} supporting sources"
            if has_supreme_court:
                reasoning += ", including Supreme Court authority"
        elif score >= 0.4:
            tier = "yellow"
            reasoning = f"Moderate confidence: {sources} source(s)"
            if sources == 1:
                reasoning += " - limited to a single authority"
        else:
            tier = "red"
            reasoning = "Low confidence"
            if has_conflicting:
                reasoning += " - conflicting authorities found"
            if sources == 0:
                reasoning += " - no direct precedent found"
            reasoning += ". Recommend consulting a qualified advocate."

        return ConfidenceAssessment(
            tier=tier,
            score=round(score, 2),
            reasoning=reasoning,
            sources_count=sources,
            conflicting_sources=has_conflicting,
        )

    def _detect_conflicting(self, results: list[dict]) -> bool:
        """Detect if results contain conflicting holdings."""
        if len(results) < 2:
            return False
        courts = [r.get("court", "") for r in results]
        return len(set(courts)) > 1 and any(r.get("is_overruled") for r in results)
