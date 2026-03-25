"""Query router that classifies queries and selects retrieval strategies.

Uses Claude Haiku to classify incoming queries into one of several financial
query types, then returns a ``QueryRouteConfig`` that tells the retrieval
pipeline how to optimise its search (top-k overrides, filters, sub-queries).
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import structlog
from pydantic import BaseModel, Field

from config.config import get_settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Query-type taxonomy
# ---------------------------------------------------------------------------

QUERY_TYPES: dict[str, str] = {
    "factual": (
        "Direct fact-lookup questions with a single, verifiable answer. "
        "Example: 'What is the current repo rate set by RBI?'"
    ),
    "analytical": (
        "Questions requiring synthesis across multiple data points or documents. "
        "Example: 'How has the SIP growth trend changed over the last 3 years?'"
    ),
    "comparison": (
        "Questions that compare two or more entities, metrics, or time periods. "
        "Example: 'Compare expense ratios of Nifty 50 index funds from HDFC and ICICI.'"
    ),
    "numerical": (
        "Questions expecting a specific number, percentage, or calculation. "
        "Example: 'What was the AUM of SBI Bluechip Fund as of December 2025?'"
    ),
    "regulatory": (
        "Questions about rules, circulars, compliance, or regulatory frameworks. "
        "Example: 'What are SEBI's new margin requirements for F&O trading?'"
    ),
}

# ---------------------------------------------------------------------------
# System prompt for Haiku classification
# ---------------------------------------------------------------------------

_CLASSIFICATION_SYSTEM_PROMPT: str = """\
You are a query classifier for a financial document retrieval system focused on \
Indian financial markets (RBI circulars, SEBI fact sheets, NSE annual reports).

Given a user query, respond with a JSON object (no markdown, no code fences) with \
these exact keys:

- "query_type": one of {query_types}
- "top_k_override": integer between 3 and 20 — how many chunks the retrieval \
  pipeline should return. Use higher values for analytical/comparison queries.
- "chunk_type_filter": either "child" or null — set to "child" for precise \
  factual/numerical lookups, null otherwise.
- "doc_type_filter": one of ["rbi_circular", "sebi_factsheet", "nse_annual_report"] \
  or null if the query is not specific to a document type.
- "run_multi_search": boolean — true if the query should be decomposed into \
  sub-queries for broader coverage (analytical and comparison queries).
- "sub_queries": list of strings — if run_multi_search is true, provide 2-4 \
  focused sub-queries that together answer the original query. Empty list otherwise.

Respond ONLY with the JSON object.
""".format(query_types=json.dumps(list(QUERY_TYPES.keys())))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class QueryRouteConfig(BaseModel):
    """Configuration produced by the query router to guide retrieval."""

    query_type: str = Field(
        description="Classified query type from the QUERY_TYPES taxonomy.",
    )
    top_k_override: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Number of chunks the retrieval pipeline should return.",
    )
    chunk_type_filter: str | None = Field(
        default=None,
        description="Restrict retrieval to 'child' chunks when precise lookup is needed.",
    )
    doc_type_filter: str | None = Field(
        default=None,
        description="Restrict retrieval to a specific document type.",
    )
    run_multi_search: bool = Field(
        default=False,
        description="Whether to decompose the query into sub-queries.",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-queries for multi-search decomposition.",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class QueryRouter:
    """Classifies a user query and produces a retrieval strategy config.

    Uses Claude Haiku (``claude-3-haiku-20240307``) for fast, low-cost
    classification.  Falls back to sensible defaults when the LLM response
    cannot be parsed.

    Parameters
    ----------
    anthropic_api_key:
        Anthropic API key.  Defaults to ``settings.ANTHROPIC_API_KEY``.
    model:
        Claude model identifier to use for classification.
    """

    _HAIKU_MODEL: str = "claude-3-haiku-20240307"

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        api_key = anthropic_api_key or settings.ANTHROPIC_API_KEY
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model or self._HAIKU_MODEL
        logger.info("query_router.init", model=self._model)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    async def classify_and_route(self, query: str) -> QueryRouteConfig:
        """Classify *query* and return a ``QueryRouteConfig``.

        The method sends the query to Claude Haiku with a structured system
        prompt and parses the JSON response.  If parsing fails, a default
        config for the ``factual`` query type is returned.

        Parameters
        ----------
        query:
            The user's natural-language query.

        Returns
        -------
        QueryRouteConfig
            Retrieval configuration derived from the classification.
        """

        logger.info("query_router.classify_start", query=query)

        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_CLASSIFICATION_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": query},
                ],
            )

            raw_text = message.content[0].text  # type: ignore[union-attr]
            parsed = self._parse_response(raw_text)

            config = QueryRouteConfig(**parsed)

            logger.info(
                "query_router.classified",
                query_type=config.query_type,
                top_k=config.top_k_override,
                multi_search=config.run_multi_search,
                sub_queries_count=len(config.sub_queries),
            )

            return config

        except Exception:
            logger.exception("query_router.classification_failed", query=query)
            return self._default_config()

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Parse the raw LLM text into a dict, tolerating minor formatting issues.

        Parameters
        ----------
        raw:
            Raw text from the Claude Haiku response.

        Returns
        -------
        dict[str, Any]
            Parsed JSON fields.

        Raises
        ------
        ValueError
            If the response cannot be parsed as JSON.
        """

        text = raw.strip()

        # Strip optional markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse Haiku response as JSON: {text!r}") from exc

        # Validate query_type
        if data.get("query_type") not in QUERY_TYPES:
            logger.warning(
                "query_router.unknown_query_type",
                received=data.get("query_type"),
            )
            data["query_type"] = "factual"

        return data

    @staticmethod
    def _default_config() -> QueryRouteConfig:
        """Return a safe fallback configuration.

        Returns
        -------
        QueryRouteConfig
            Default config for ``factual`` query type.
        """

        return QueryRouteConfig(
            query_type="factual",
            top_k_override=5,
            chunk_type_filter="child",
            doc_type_filter=None,
            run_multi_search=False,
            sub_queries=[],
        )
