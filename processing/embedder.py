"""Dense embedding service wrapping sentence-transformers.

Provides a thin, batch-aware interface for generating embeddings that are
stored in Qdrant and used for dense retrieval.
"""

from __future__ import annotations

import structlog
from sentence_transformers import SentenceTransformer

from findocs.config.config import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Generate dense vector embeddings using a SentenceTransformer model.

    The model is loaded once at construction time and reused for all
    subsequent calls.  Texts are processed in batches of 64 to balance
    throughput against memory usage.
    """

    _DEFAULT_BATCH_SIZE: int = 64

    def __init__(self, *, model_name: str | None = None, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        """Load the SentenceTransformer model.

        Args:
            model_name: HuggingFace model identifier.  When *None* the value
                is read from ``settings.EMBEDDING_MODEL``.
            batch_size: Number of texts encoded in a single forward pass.
        """
        settings = get_settings()
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._batch_size = batch_size

        logger.info("embedding_service_loading_model", model=self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._embedding_dim: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        logger.info(
            "embedding_service_ready",
            model=self._model_name,
            dim=self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their dense vectors.

        Long input lists are split into batches of ``self._batch_size`` to
        prevent out-of-memory errors on resource-constrained machines.

        Args:
            texts: Texts to embed.  Empty strings are permitted but will
                produce zero-ish vectors depending on the model.

        Returns:
            List of float vectors, one per input text.  Each vector has
            length ``embedding_dim``.
        """
        if not texts:
            return []

        logger.debug("embed_texts_start", count=len(texts), batch_size=self._batch_size)

        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            batch_vectors = self._model.encode(
                batch, show_progress_bar=False, convert_to_numpy=True
            )
            all_embeddings.extend(batch_vectors.tolist())

        logger.debug("embed_texts_done", count=len(all_embeddings))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        This is a convenience wrapper around :meth:`embed_texts` that returns
        a single vector instead of a list.

        Args:
            query: The search query text.

        Returns:
            Dense float vector for the query.
        """
        logger.debug("embed_query", query_length=len(query))
        vector = self._model.encode(query, show_progress_bar=False, convert_to_numpy=True)
        return vector.tolist()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the vectors produced by the loaded model."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Name / identifier of the loaded model."""
        return self._model_name
