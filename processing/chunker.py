"""Semantic and parent-child chunking for parsed financial documents.

This module splits parsed documents into semantically coherent chunks using
embedding similarity to detect topic boundaries.  Each parent chunk (~512 tokens)
is further divided into overlapping child chunks (~128 tokens) so that retrieval
operates on fine-grained units while the LLM receives the broader parent context.
"""

from __future__ import annotations

import uuid
from typing import Literal

import numpy as np
import spacy
import structlog
import tiktoken
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from findocs.ingestion.models import ParsedDocument

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model for a single chunk
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A single chunk derived from a parsed financial document."""

    chunk_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for this chunk.")
    parent_id: uuid.UUID | None = Field(default=None, description="ID of the parent chunk if this is a child.")
    content: str = Field(..., description="Text content of the chunk.")
    chunk_level: Literal["parent", "child"] = Field(..., description="Whether this is a parent or child chunk.")
    chunk_type: Literal["text", "table", "chart", "mixed"] = Field(
        default="text", description="Content type present in the chunk."
    )
    doc_source: str = Field(default="", description="Source file path or URL of the originating document.")
    doc_type: str = Field(default="", description="Document type (e.g. rbi_circular, sebi_factsheet).")
    doc_date: str | None = Field(default=None, description="Publication date of the source document (ISO format).")
    page_num: int | None = Field(default=None, description="Page number in the original document.")
    headings_context: str = Field(
        default="", description="Concatenated ancestor headings that give this chunk context."
    )
    token_count: int = Field(default=0, description="Number of tokens (cl100k_base) in the chunk content.")
    char_count: int = Field(default=0, description="Number of characters in the chunk content.")


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """Splits a *ParsedDocument* into semantically coherent parent/child chunks.

    # ARCHITECTURE DECISION: Why semantic chunking over fixed-size
    # Fixed-size chunking blindly splits at token boundaries, which frequently breaks
    # mid-sentence or mid-paragraph in financial documents. Semantic chunking uses
    # embedding similarity between adjacent sentences to find natural topic boundaries,
    # preserving domain-specific context (e.g., keeping an entire policy clause together).
    # The parent-child hierarchy allows retrieval on fine-grained child chunks (128 tokens)
    # while returning the broader parent chunk (512 tokens) as context to the LLM.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        *,
        parent_max_tokens: int = 512,
        child_max_tokens: int = 128,
        child_overlap_tokens: int = 20,
        similarity_threshold: float = 0.6,
    ) -> None:
        """Initialise the chunker.

        Args:
            embedding_model: A loaded SentenceTransformer used to compute
                sentence embeddings for boundary detection.
            parent_max_tokens: Target maximum token count for parent chunks.
            child_max_tokens: Target maximum token count for child chunks.
            child_overlap_tokens: Number of overlapping tokens between
                consecutive child chunks.
            similarity_threshold: Cosine-similarity threshold below which a
                semantic boundary is declared between adjacent sentences.
        """
        self._embedding_model = embedding_model
        self._parent_max_tokens = parent_max_tokens
        self._child_max_tokens = child_max_tokens
        self._child_overlap_tokens = child_overlap_tokens
        self._similarity_threshold = similarity_threshold

        self._tokenizer = tiktoken.get_encoding("cl100k_base")

        # Load spaCy model for sentence segmentation
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spacy_model_missing", model="en_core_web_sm")
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed.  "
                "Run: python -m spacy download en_core_web_sm"
            )

        # Increase the max_length so very large documents are not rejected
        self._nlp.max_length = 5_000_000

        logger.info(
            "semantic_chunker_init",
            parent_max_tokens=parent_max_tokens,
            child_max_tokens=child_max_tokens,
            child_overlap_tokens=child_overlap_tokens,
            similarity_threshold=similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """Chunk a full parsed document into parent and child chunks.

        Processing pipeline:
            1. Concatenate blocks with type markers for tables/charts.
            2. Split concatenated text into sentences via spaCy.
            3. Compute sentence embeddings.
            4. Detect semantic boundaries (cosine sim < threshold).
            5. Group sentences into parent chunks (~512 tokens).
            6. Split each parent into overlapping child chunks (~128 tokens).

        Args:
            doc: A fully parsed financial document.

        Returns:
            A flat list of *Chunk* objects containing both parent and child
            chunks.  Each child's ``parent_id`` points to its parent.
        """
        log = logger.bind(source=str(doc.source_path), doc_type=doc.doc_type)
        log.info("chunk_document_start", num_blocks=len(doc.blocks))

        # 1. Concatenate blocks with type markers --------------------------
        annotated_parts: list[str] = []
        block_page_map: list[int] = []  # tracks page_num for each sentence
        block_type_set: set[str] = set()

        for block in doc.blocks:
            block_type_set.add(block.block_type)
            if block.block_type == "table":
                annotated_parts.append(f"[TABLE]\n{block.content}\n[/TABLE]")
            elif block.block_type == "chart":
                annotated_parts.append(f"[CHART DESCRIPTION]\n{block.content}\n[/CHART]")
            else:
                annotated_parts.append(block.content)
            block_page_map.append(block.page_num)

        full_text = "\n\n".join(annotated_parts)

        if not full_text.strip():
            log.warning("chunk_document_empty")
            return []

        # 2. Sentence segmentation -----------------------------------------
        spacy_doc = self._nlp(full_text)
        sentences: list[str] = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]

        if not sentences:
            log.warning("chunk_document_no_sentences")
            return []

        log.debug("sentences_extracted", count=len(sentences))

        # 3. Compute sentence embeddings -----------------------------------
        embeddings: np.ndarray = self._embedding_model.encode(
            sentences, show_progress_bar=False, convert_to_numpy=True
        )

        # 4. Find semantic boundaries --------------------------------------
        boundaries = self._find_semantic_boundaries(embeddings, self._similarity_threshold)
        log.debug("semantic_boundaries_found", count=len(boundaries))

        # 5. Group sentences into semantic segments, then build parents -----
        segments = self._group_by_boundaries(sentences, boundaries)

        # Build a mapping from sentence -> approximate page_num
        sentence_page: list[int] = self._map_sentences_to_pages(full_text, sentences, doc)

        parent_chunks: list[Chunk] = []
        all_chunks: list[Chunk] = []

        for segment_sentences in segments:
            segment_text = " ".join(segment_sentences)
            parent_pieces = self._split_into_tokens(
                segment_text, self._parent_max_tokens, overlap=0
            )

            for piece in parent_pieces:
                piece_token_count = len(self._tokenizer.encode(piece))
                dominant_page = self._dominant_page(piece, sentence_page, sentences)
                chunk_type = self._determine_chunk_type(piece, block_type_set)

                parent = Chunk(
                    content=piece,
                    chunk_level="parent",
                    chunk_type=chunk_type,
                    page_num=dominant_page,
                    token_count=piece_token_count,
                    char_count=len(piece),
                )
                parent_chunks.append(parent)
                all_chunks.append(parent)

        # 6. Create child chunks for each parent ---------------------------
        for parent in parent_chunks:
            child_texts = self._split_into_tokens(
                parent.content, self._child_max_tokens, self._child_overlap_tokens
            )
            for child_text in child_texts:
                child_token_count = len(self._tokenizer.encode(child_text))
                child = Chunk(
                    parent_id=parent.chunk_id,
                    content=child_text,
                    chunk_level="child",
                    chunk_type=parent.chunk_type,
                    page_num=parent.page_num,
                    token_count=child_token_count,
                    char_count=len(child_text),
                )
                all_chunks.append(child)

        log.info(
            "chunk_document_done",
            parents=len(parent_chunks),
            total_chunks=len(all_chunks),
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_semantic_boundaries(
        self, sentence_embeddings: np.ndarray, threshold: float = 0.6
    ) -> list[int]:
        """Return sentence indices where a semantic boundary is detected.

        A boundary is placed *after* sentence ``i`` when the cosine similarity
        between the embeddings of sentence ``i`` and sentence ``i + 1`` falls
        below *threshold*.

        Args:
            sentence_embeddings: 2-D array of shape ``(n_sentences, dim)``.
            threshold: Similarity below which a boundary is declared.

        Returns:
            Sorted list of boundary indices (each is a sentence index after
            which a break should occur).
        """
        if len(sentence_embeddings) < 2:
            return []

        boundaries: list[int] = []
        for i in range(len(sentence_embeddings) - 1):
            vec_a = sentence_embeddings[i]
            vec_b = sentence_embeddings[i + 1]
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                # Degenerate zero vector -> treat as boundary
                boundaries.append(i)
                continue
            cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            if cosine_sim < threshold:
                boundaries.append(i)

        return boundaries

    def _split_into_tokens(
        self, text: str, max_tokens: int, overlap: int = 0
    ) -> list[str]:
        """Split *text* into pieces of at most *max_tokens* with optional overlap.

        Uses the ``tiktoken`` ``cl100k_base`` encoding for tokenisation.

        Args:
            text: The input text to split.
            max_tokens: Maximum number of tokens per piece.
            overlap: Number of overlapping tokens between consecutive pieces.

        Returns:
            List of text pieces, each decoded back from the token ids.
        """
        tokens = self._tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        pieces: list[str] = []
        start = 0
        step = max(max_tokens - overlap, 1)

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            piece_tokens = tokens[start:end]
            piece_text = self._tokenizer.decode(piece_tokens)
            if piece_text.strip():
                pieces.append(piece_text)
            if end >= len(tokens):
                break
            start += step

        return pieces

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_boundaries(
        sentences: list[str], boundaries: list[int]
    ) -> list[list[str]]:
        """Group sentences into segments split at the given boundary indices.

        Args:
            sentences: Ordered list of sentence strings.
            boundaries: Indices after which a split should occur.

        Returns:
            List of sentence groups (each a list of strings).
        """
        if not boundaries:
            return [sentences]

        segments: list[list[str]] = []
        prev = 0
        for b in sorted(boundaries):
            split_idx = b + 1
            if split_idx > prev:
                segments.append(sentences[prev:split_idx])
            prev = split_idx
        if prev < len(sentences):
            segments.append(sentences[prev:])
        return segments

    def _map_sentences_to_pages(
        self,
        full_text: str,
        sentences: list[str],
        doc: ParsedDocument,
    ) -> list[int]:
        """Build a best-effort mapping from each sentence to a page number.

        The heuristic assigns each sentence the page number of the last block
        whose content appears before or at the sentence's position in
        *full_text*.

        Args:
            full_text: The fully concatenated document text.
            sentences: Ordered list of sentences extracted from *full_text*.
            doc: The original parsed document (used for page numbers).

        Returns:
            List of page numbers aligned with *sentences*.
        """
        # Build character-offset -> page_num mapping from blocks
        offsets: list[tuple[int, int]] = []  # (char_offset, page_num)
        search_start = 0
        for block in doc.blocks:
            idx = full_text.find(block.content, search_start)
            if idx != -1:
                offsets.append((idx, block.page_num))
                search_start = idx

        if not offsets:
            default_page = doc.blocks[0].page_num if doc.blocks else 1
            return [default_page] * len(sentences)

        sentence_pages: list[int] = []
        for sent in sentences:
            sent_idx = full_text.find(sent)
            page = offsets[0][1]
            for offset, pnum in offsets:
                if offset <= (sent_idx if sent_idx != -1 else 0):
                    page = pnum
                else:
                    break
            sentence_pages.append(page)

        return sentence_pages

    def _dominant_page(
        self,
        piece_text: str,
        sentence_page: list[int],
        sentences: list[str],
    ) -> int:
        """Return the most frequent page number among sentences in *piece_text*.

        Args:
            piece_text: Text of the chunk piece.
            sentence_page: Page number for each sentence.
            sentences: Full list of sentences.

        Returns:
            The page number that appears most often; falls back to 1.
        """
        page_counts: dict[int, int] = {}
        for sent, page in zip(sentences, sentence_page):
            if sent in piece_text:
                page_counts[page] = page_counts.get(page, 0) + 1
        if not page_counts:
            return 1
        return max(page_counts, key=page_counts.get)  # type: ignore[arg-type]

    @staticmethod
    def _determine_chunk_type(
        text: str, block_types: set[str]
    ) -> Literal["text", "table", "chart", "mixed"]:
        """Infer the chunk type from marker tags present in *text*.

        Args:
            text: The chunk text (may contain ``[TABLE]`` / ``[CHART ...]``
                markers).
            block_types: Set of block types found in the source document.

        Returns:
            One of ``"text"``, ``"table"``, ``"chart"``, or ``"mixed"``.
        """
        has_table = "[TABLE]" in text
        has_chart = "[CHART DESCRIPTION]" in text
        if has_table and has_chart:
            return "mixed"
        if has_table:
            return "table"
        if has_chart:
            return "chart"
        return "text"
