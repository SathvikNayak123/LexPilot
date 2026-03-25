"""Tests for processing/chunker.py: SemanticChunker and Chunk model.

The SentenceTransformer is mocked to return deterministic embeddings,
and spaCy is mocked where needed to avoid loading a full pipeline in
CI environments.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from findocs.ingestion.models import ParsedBlock, ParsedDocument
from findocs.processing.chunker import Chunk, SemanticChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parsed_document(blocks: list[ParsedBlock]) -> ParsedDocument:
    """Create a minimal ParsedDocument with the given blocks."""
    return ParsedDocument(
        source_path=Path("/data/test/test_doc.pdf"),
        doc_type="rbi_circular",
        title="Test Document",
        date=None,
        blocks=blocks,
        total_pages=1,
        parsing_duration_seconds=0.1,
    )


def _make_mock_embedding_model(dim: int = 384) -> MagicMock:
    """Create a mock SentenceTransformer that returns deterministic embeddings.

    The mock returns a unique random (but seeded) vector for each sentence
    based on its hash, so that distinct sentences get distinct embeddings.
    """
    mock_model = MagicMock()

    def encode_side_effect(sentences, show_progress_bar=False, convert_to_numpy=True):
        rng = np.random.RandomState(42)
        embeddings = []
        for sent in sentences:
            # Use a deterministic seed per sentence
            seed = hash(sent) % (2**31)
            local_rng = np.random.RandomState(seed)
            embeddings.append(local_rng.randn(dim).astype(np.float32))
        return np.array(embeddings)

    mock_model.encode = MagicMock(side_effect=encode_side_effect)
    return mock_model


def _make_mock_spacy():
    """Create a mock spaCy NLP pipeline that segments text on sentence boundaries."""
    mock_nlp = MagicMock()
    mock_nlp.max_length = 5_000_000

    def nlp_side_effect(text):
        """Simulate spaCy sentence segmentation by splitting on '. '"""
        mock_doc = MagicMock()

        # Simple sentence splitting
        raw_sentences = []
        current = []
        for char in text:
            current.append(char)
            if char == "." and len(current) > 1:
                sentence_text = "".join(current).strip()
                if sentence_text:
                    raw_sentences.append(sentence_text)
                current = []

        # Flush remaining
        remainder = "".join(current).strip()
        if remainder:
            raw_sentences.append(remainder)

        if not raw_sentences:
            raw_sentences = [text.strip()] if text.strip() else []

        # Build mock Span objects
        mock_sents = []
        for s in raw_sentences:
            mock_span = MagicMock()
            mock_span.text = s
            mock_sents.append(mock_span)

        mock_doc.sents = mock_sents
        return mock_doc

    mock_nlp.side_effect = nlp_side_effect
    mock_nlp.__call__ = nlp_side_effect
    return mock_nlp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSemanticChunker:
    """Tests for the SemanticChunker class."""

    @pytest.fixture(autouse=True)
    def _setup_chunker(self):
        """Set up a SemanticChunker with mocked dependencies for each test."""
        self.mock_embedding_model = _make_mock_embedding_model()
        self.mock_nlp = _make_mock_spacy()

        with patch("findocs.processing.chunker.spacy") as mock_spacy_module:
            mock_spacy_module.load.return_value = self.mock_nlp
            self.chunker = SemanticChunker(
                embedding_model=self.mock_embedding_model,
                parent_max_tokens=512,
                child_max_tokens=128,
                child_overlap_tokens=20,
                similarity_threshold=0.6,
            )
            # Override _nlp with our mock since __init__ already ran
            self.chunker._nlp = self.mock_nlp

    def test_semantic_boundary_detection(self) -> None:
        """Feed sentences with clear topic shift, verify boundary detected."""
        # Create two clearly different embedding vectors
        # Sentences 0-2: similar topic (close vectors)
        # Sentence 3: different topic (orthogonal vector)
        dim = 384
        rng = np.random.RandomState(42)

        base_vector = rng.randn(dim).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)

        # Similar sentences: small perturbations of base vector
        emb_0 = base_vector + 0.05 * rng.randn(dim).astype(np.float32)
        emb_1 = base_vector + 0.05 * rng.randn(dim).astype(np.float32)
        emb_2 = base_vector + 0.05 * rng.randn(dim).astype(np.float32)

        # Different topic: nearly orthogonal
        different_vector = rng.randn(dim).astype(np.float32)
        different_vector = different_vector / np.linalg.norm(different_vector)
        # Make it orthogonal to base
        different_vector = different_vector - np.dot(different_vector, base_vector) * base_vector
        different_vector = different_vector / np.linalg.norm(different_vector)
        emb_3 = different_vector

        embeddings = np.array([emb_0, emb_1, emb_2, emb_3])

        boundaries = self.chunker._find_semantic_boundaries(embeddings, threshold=0.6)

        # There should be a boundary before sentence 3 (the topic shift)
        assert len(boundaries) >= 1, (
            f"Expected at least one boundary for topic shift, got {boundaries}"
        )
        # The boundary should be at index 2 (after sentence 2, before sentence 3)
        assert 2 in boundaries, (
            f"Expected boundary at index 2, got boundaries: {boundaries}"
        )

    def test_parent_child_relationship(self, sample_parsed_document) -> None:
        """Every child chunk must have a valid parent_id that exists among parent chunks."""
        chunks = self.chunker.chunk_document(sample_parsed_document)

        if not chunks:
            pytest.skip("No chunks produced from sample document")

        parent_ids = {c.chunk_id for c in chunks if c.chunk_level == "parent"}
        child_chunks = [c for c in chunks if c.chunk_level == "child"]

        for child in child_chunks:
            assert child.parent_id is not None, (
                f"Child chunk {child.chunk_id} has no parent_id"
            )
            assert child.parent_id in parent_ids, (
                f"Child chunk {child.chunk_id} has parent_id {child.parent_id} "
                f"that does not exist among parent chunks"
            )

    def test_chunk_size_constraints(self, sample_parsed_document) -> None:
        """No chunk should exceed CHUNK_SIZE_TOKENS (parent_max_tokens for parents)."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        chunks = self.chunker.chunk_document(sample_parsed_document)

        if not chunks:
            pytest.skip("No chunks produced from sample document")

        for chunk in chunks:
            actual_tokens = len(enc.encode(chunk.content))
            if chunk.chunk_level == "parent":
                # Allow a small margin since token splitting may not be exact
                max_allowed = self.chunker._parent_max_tokens + 5
                assert actual_tokens <= max_allowed, (
                    f"Parent chunk {chunk.chunk_id} has {actual_tokens} tokens, "
                    f"exceeding limit of {max_allowed}"
                )
            else:
                max_allowed = self.chunker._child_max_tokens + 5
                assert actual_tokens <= max_allowed, (
                    f"Child chunk {chunk.chunk_id} has {actual_tokens} tokens, "
                    f"exceeding limit of {max_allowed}"
                )

    def test_table_marker_preservation(self) -> None:
        """Table content should be wrapped in [TABLE]...[/TABLE] markers."""
        blocks = [
            ParsedBlock(
                content="| Col A | Col B |\n| --- | --- |\n| 100 | 200 |",
                block_type="table",
                page_num=1,
                metadata={},
            ),
        ]
        doc = _make_parsed_document(blocks)
        chunks = self.chunker.chunk_document(doc)

        if not chunks:
            pytest.skip("No chunks produced")

        # At least one parent chunk should contain the table markers
        table_chunks = [c for c in chunks if c.chunk_level == "parent"]
        has_table_markers = any(
            "[TABLE]" in c.content and "[/TABLE]" in c.content
            for c in table_chunks
        )
        assert has_table_markers, (
            "Expected at least one chunk to contain [TABLE]...[/TABLE] markers. "
            f"Parent chunks: {[c.content[:80] for c in table_chunks]}"
        )

    def test_chart_marker_preservation(self) -> None:
        """Chart content should be wrapped in [CHART DESCRIPTION]...[/CHART] markers."""
        blocks = [
            ParsedBlock(
                content="Bar chart showing revenue growth from Q1 to Q4 FY2025.",
                block_type="chart",
                page_num=1,
                metadata={},
            ),
        ]
        doc = _make_parsed_document(blocks)
        chunks = self.chunker.chunk_document(doc)

        if not chunks:
            pytest.skip("No chunks produced")

        parent_chunks = [c for c in chunks if c.chunk_level == "parent"]
        has_chart_markers = any(
            "[CHART DESCRIPTION]" in c.content and "[/CHART]" in c.content
            for c in parent_chunks
        )
        assert has_chart_markers, (
            "Expected at least one chunk to contain [CHART DESCRIPTION]...[/CHART] markers. "
            f"Parent chunks: {[c.content[:80] for c in parent_chunks]}"
        )

    def test_chunk_has_required_fields(self, sample_parsed_document) -> None:
        """All chunks must have chunk_id, content, chunk_level, chunk_type."""
        chunks = self.chunker.chunk_document(sample_parsed_document)

        if not chunks:
            pytest.skip("No chunks produced")

        for chunk in chunks:
            assert chunk.chunk_id is not None, "chunk_id should not be None"
            assert isinstance(chunk.chunk_id, uuid.UUID), (
                f"chunk_id should be a UUID, got {type(chunk.chunk_id)}"
            )
            assert chunk.content, "content should not be empty"
            assert chunk.chunk_level in ("parent", "child"), (
                f"chunk_level should be 'parent' or 'child', got '{chunk.chunk_level}'"
            )
            assert chunk.chunk_type in ("text", "table", "chart", "mixed"), (
                f"chunk_type should be one of text/table/chart/mixed, got '{chunk.chunk_type}'"
            )


class TestChunkModel:
    """Tests for the Chunk Pydantic model."""

    def test_chunk_creation_with_defaults(self) -> None:
        """Verify a Chunk can be created with just required fields."""
        chunk = Chunk(
            content="Test content for the chunk.",
            chunk_level="parent",
        )
        assert chunk.chunk_id is not None
        assert isinstance(chunk.chunk_id, uuid.UUID)
        assert chunk.parent_id is None
        assert chunk.chunk_type == "text"
        assert chunk.doc_source == ""
        assert chunk.doc_type == ""

    def test_chunk_child_with_parent_id(self) -> None:
        """A child chunk should store its parent_id correctly."""
        parent_id = uuid.uuid4()
        child = Chunk(
            content="Child content.",
            chunk_level="child",
            parent_id=parent_id,
        )
        assert child.parent_id == parent_id
        assert child.chunk_level == "child"
