"""BM25 sparse-vector generation for hybrid search in Qdrant.

This module builds a BM25 index over a text corpus and converts BM25 scores
into Qdrant-compatible ``SparseVector`` objects (token-index / score pairs)
for sparse retrieval.
"""

from __future__ import annotations

import re

import structlog
from qdrant_client.models import SparseVector
from rank_bm25 import BM25Okapi

logger = structlog.get_logger(__name__)


class BM25Indexer:
    """Fit a BM25 model on a corpus and produce sparse vectors for Qdrant.

    The indexer tokenises documents with a lightweight regex tokeniser,
    fits a ``BM25Okapi`` instance, and exposes methods to generate
    ``SparseVector`` objects suitable for Qdrant's sparse-vector index.
    """

    def __init__(self, corpus: list[str] | None = None) -> None:
        """Optionally fit the BM25 model immediately on *corpus*.

        Args:
            corpus: If provided, :meth:`fit` is called automatically.
                Pass *None* to defer fitting.
        """
        self._bm25: BM25Okapi | None = None
        self._vocab: dict[str, int] = {}
        self._is_fitted: bool = False

        if corpus is not None:
            self.fit(corpus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, corpus: list[str]) -> None:
        """Tokenise *corpus* and fit the BM25 model.

        Args:
            corpus: List of document strings to index.

        Raises:
            ValueError: If *corpus* is empty.
        """
        if not corpus:
            raise ValueError("Cannot fit BM25 on an empty corpus.")

        logger.info("bm25_fit_start", corpus_size=len(corpus))

        tokenised = [self._tokenise(doc) for doc in corpus]

        # Build vocabulary mapping (token -> unique integer index)
        self._vocab = {}
        idx = 0
        for tokens in tokenised:
            for token in tokens:
                if token not in self._vocab:
                    self._vocab[token] = idx
                    idx += 1

        self._bm25 = BM25Okapi(tokenised)
        self._is_fitted = True

        logger.info("bm25_fit_done", vocab_size=len(self._vocab))

    def get_sparse_vector(self, text: str) -> SparseVector:
        """Return a Qdrant ``SparseVector`` with BM25 scores for *text*.

        Each unique token in *text* that exists in the fitted vocabulary
        produces one entry in the sparse vector (index, value).

        Args:
            text: Input text to vectorise.

        Returns:
            A ``SparseVector`` with non-zero BM25 score entries.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._assert_fitted()

        tokens = self._tokenise(text)
        scores = self._bm25.get_scores(tokens)  # type: ignore[union-attr]

        indices: list[int] = []
        values: list[float] = []

        # Collect unique tokens that appear in both text and vocab
        seen_tokens: set[str] = set()
        for token in tokens:
            if token in self._vocab and token not in seen_tokens:
                seen_tokens.add(token)
                token_idx = self._vocab[token]
                score = float(scores[token_idx]) if token_idx < len(scores) else 0.0
                if score > 0.0:
                    indices.append(token_idx)
                    values.append(score)

        # Fall-back: if no scores survived, produce an "empty" sparse vector
        # with a single zero entry so Qdrant doesn't reject the point.
        if not indices:
            indices = [0]
            values = [0.0]

        return SparseVector(indices=indices, values=values)

    def batch_get_sparse_vectors(self, texts: list[str]) -> list[SparseVector]:
        """Return sparse vectors for a batch of texts.

        This is a convenience method that calls :meth:`get_sparse_vector` for
        each text.

        Args:
            texts: List of texts to vectorise.

        Returns:
            List of ``SparseVector`` objects, one per input text.
        """
        self._assert_fitted()

        logger.debug("bm25_batch_start", count=len(texts))
        vectors = [self.get_sparse_vector(t) for t in texts]
        logger.debug("bm25_batch_done", count=len(vectors))
        return vectors

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the BM25 model has been fitted on a corpus."""
        return self._is_fitted

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the fitted vocabulary."""
        return len(self._vocab)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lowercase and split *text* on non-alphanumeric characters.

        Args:
            text: Raw input string.

        Returns:
            List of lowercase tokens.
        """
        return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1]

    def _assert_fitted(self) -> None:
        """Raise if the model has not been fitted yet.

        Raises:
            RuntimeError: When :meth:`fit` has not been called.
        """
        if not self._is_fitted or self._bm25 is None:
            raise RuntimeError(
                "BM25Indexer has not been fitted. Call fit(corpus) first."
            )
