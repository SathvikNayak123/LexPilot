from pydantic import BaseModel
from typing import Literal


class LegalRouteDecision(BaseModel):
    tier: Literal[1, 2, 3]
    complexity_score: int
    reasoning: str
    force_tier3: bool = False  # Domain override


class LegalModelRouter:
    """Routes legal queries to appropriate model tier.
    Legal-domain override: compliance reports and precedent analysis
    ALWAYS route to Tier 3 regardless of apparent simplicity."""

    TIER1_PATTERNS = [
        "what is section", "lookup section", "show me section",
        "dpdp section", "define ",
    ]

    # These ALWAYS go to Tier 3 — cost of wrong answer is too high
    FORCE_TIER3_KEYWORDS = [
        "compliance", "audit", "dpdp", "precedent", "case law",
        "overruled", "binding authority", "ratio decidendi",
    ]

    def route(self, query: str) -> LegalRouteDecision:
        query_lower = query.lower().strip()

        # Check Tier 1 (direct lookup, no LLM)
        for pattern in self.TIER1_PATTERNS:
            if query_lower.startswith(pattern):
                return LegalRouteDecision(
                    tier=1, complexity_score=1,
                    reasoning="Direct section lookup",
                )

        # Check forced Tier 3 (high-stakes legal tasks)
        for kw in self.FORCE_TIER3_KEYWORDS:
            if kw in query_lower:
                return LegalRouteDecision(
                    tier=3, complexity_score=8,
                    reasoning=f"Legal domain override: '{kw}' requires highest accuracy",
                    force_tier3=True,
                )

        # Default scoring
        score = 4  # Base
        if len(query.split()) > 25:
            score += 1
        if "compare" in query_lower or "analyze" in query_lower:
            score += 2
        if "and" in query_lower:  # Multi-part query
            score += 1

        tier = 2 if score <= 5 else 3
        return LegalRouteDecision(tier=tier, complexity_score=score, reasoning="Complexity-based routing")
