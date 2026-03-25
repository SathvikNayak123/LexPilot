"""Cross-encoder reranker for FinDocs retrieval pipeline.

Loads a cross-encoder model (default: ``cross-encoder/ms-marco-MiniLM-L-6-v2``)
and rescores ``(query, chunk)`` pairs to improve ranking quality after the
initial dense + sparse retrieval stage.
"""

from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

from config.config import get_settings
from retrieval.qdrant_client import SearchResult

logger = structlog.get_logger(__name__)


class Reranker:
    """Cross-encoder reranker that rescores search results.

    The model is loaded once during ``__init__`` and reused across calls.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the cross-encoder.  Falls back to
        ``settings.RERANKER_MODEL`` when not provided.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name: str = model_name or settings.RERANKER_MODEL
        logger.info("reranker.loading_model", model=self._model_name)
        self._model: CrossEncoder = CrossEncoder(self._model_name)
        logger.info("reranker.model_loaded", model=self._model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results using the cross-encoder.

        Each ``(query, result.content)`` pair is scored by the cross-encoder.
        Results are sorted by descending cross-encoder score and the top-*k*
        are returned.  The ``score`` field on each returned ``SearchResult`` is
        replaced with the cross-encoder score so downstream consumers can use a
        single ranking value.

        Parameters
        ----------
        query:
            The user's natural-language query.
        results:
            Candidate ``SearchResult`` objects from the retrieval stage.
        top_k:
            Number of top results to return after reranking.

        Returns
        -------
        list[SearchResult]
            The *top_k* results sorted by cross-encoder score (descending).
        """

        if not results:
            logger.warning("reranker.empty_input")
            return []

        pairs: list[list[str]] = [[query, r.content] for r in results]

        scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[union-attr]

        scored_results: list[tuple[float, SearchResult]] = []
        for score, result in zip(scores, results, strict=True):
            updated = result.model_copy(update={"score": float(score)})
            scored_results.append((float(score), updated))

        scored_results.sort(key=lambda t: t[0], reverse=True)

        top_results = [sr for _, sr in scored_results[:top_k]]

        logger.info(
            "reranker.complete",
            input_count=len(results),
            output_count=len(top_results),
            top_score=scored_results[0][0] if scored_results else None,
        )

        return top_results
