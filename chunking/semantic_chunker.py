import spacy
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Optional
import structlog

from config.config import settings
from ingestion.models import ParsedDocument, ParsedBlock
from chunking.models import ChildChunk, ParentChunk, ChunkMetadata

logger = structlog.get_logger()


class SemanticChunker:
    """Semantic chunking with parent-child hierarchy for legal documents."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.semantic_threshold = settings.semantic_threshold
        self.parent_target_tokens = settings.parent_chunk_tokens
        self.child_target_tokens = settings.child_chunk_tokens
        self.child_overlap_tokens = settings.chunk_overlap_tokens

    def chunk_document(self, doc: ParsedDocument) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Chunk a parsed document into parent and child chunks."""
        # Step 1: Extract full text with heading context
        text_with_headings = self._extract_text_with_headings(doc.blocks)

        # Step 2: Sentence segmentation
        sentences = self._segment_sentences(text_with_headings)
        if not sentences:
            return [], []

        # Step 3: Semantic splitting into parent chunks
        parent_texts = self._semantic_split(sentences)

        # Step 4: Build parent and child chunks
        parents = []
        children = []

        for i, parent_text in enumerate(parent_texts):
            heading_ctx = self._find_nearest_heading(parent_text, doc.blocks)

            parent = ParentChunk(
                id=f"{doc.document_id}_p{i}",
                document_id=doc.document_id,
                content=parent_text,
                token_count=len(self.encoder.encode(parent_text)),
                char_count=len(parent_text),
                metadata=ChunkMetadata(
                    document_id=doc.document_id,
                    doc_type=doc.doc_type,
                    citation=doc.citation,
                    source=doc.source,
                    court=doc.court,
                    date=str(doc.date) if doc.date else None,
                    heading_context=heading_ctx,
                    chunk_index=i,
                ),
            )

            # Split parent into children
            child_texts = self._split_into_children(parent_text)
            child_ids = []

            for j, child_text in enumerate(child_texts):
                child_id = f"{doc.document_id}_p{i}_c{j}"
                child_ids.append(child_id)
                children.append(ChildChunk(
                    id=child_id,
                    parent_id=parent.id,
                    content=child_text,
                    token_count=len(self.encoder.encode(child_text)),
                    char_count=len(child_text),
                    metadata=parent.metadata.model_copy(),
                ))

            parent.child_ids = child_ids
            parents.append(parent)

        logger.info("document_chunked", doc_id=doc.document_id,
                     parents=len(parents), children=len(children))
        return parents, children

    def _extract_text_with_headings(self, blocks: list[ParsedBlock]) -> str:
        """Concatenate blocks preserving heading markers."""
        parts = []
        for block in blocks:
            if block.is_heading:
                parts.append(f"\n## {block.content}\n")
            else:
                parts.append(block.content)
        return "\n".join(parts)

    def _segment_sentences(self, text: str) -> list[str]:
        """Use spaCy for accurate sentence segmentation."""
        # Process in chunks to handle long documents
        max_chars = 100000  # spaCy limit
        sentences = []
        for i in range(0, len(text), max_chars):
            chunk = text[i:i + max_chars]
            doc = self.nlp(chunk)
            sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
        return sentences

    def _semantic_split(self, sentences: list[str]) -> list[str]:
        """Split sentences into semantic groups using embedding similarity."""
        if len(sentences) <= 1:
            return [" ".join(sentences)]

        # Embed all sentences
        embeddings = self.embedder.encode(sentences, show_progress_bar=False)

        # Compute consecutive similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)

        # Find split points where similarity drops below threshold
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < self.semantic_threshold:
                split_indices.append(i + 1)

        # Build parent chunks from sentence groups
        parent_texts = []
        for idx in range(len(split_indices)):
            start = split_indices[idx]
            end = split_indices[idx + 1] if idx + 1 < len(split_indices) else len(sentences)
            group = " ".join(sentences[start:end])

            # If group exceeds target parent size, force-split by token count
            if len(self.encoder.encode(group)) > self.parent_target_tokens * 1.5:
                sub_groups = self._force_split_by_tokens(sentences[start:end], self.parent_target_tokens)
                parent_texts.extend(sub_groups)
            else:
                parent_texts.append(group)

        return parent_texts

    def _split_into_children(self, parent_text: str) -> list[str]:
        """Split parent chunk into overlapping child chunks."""
        tokens = self.encoder.encode(parent_text)

        if len(tokens) <= self.child_target_tokens:
            return [parent_text]

        children = []
        start = 0
        while start < len(tokens):
            end = min(start + self.child_target_tokens, len(tokens))
            child_tokens = tokens[start:end]
            children.append(self.encoder.decode(child_tokens))
            start += self.child_target_tokens - self.child_overlap_tokens

        return children

    def _force_split_by_tokens(self, sentences: list[str], target: int) -> list[str]:
        """Force-split a sentence group by token count."""
        groups = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(self.encoder.encode(sent))
            if current_tokens + sent_tokens > target and current:
                groups.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            groups.append(" ".join(current))
        return groups

    def _find_nearest_heading(self, chunk_text: str, blocks: list[ParsedBlock]) -> str:
        """Find the nearest heading above this chunk's content."""
        for block in reversed(blocks):
            if block.is_heading and block.content in chunk_text[:200]:
                return block.content
        # Fallback: find any heading that precedes the chunk start
        chunk_start = chunk_text[:50]
        last_heading = ""
        for block in blocks:
            if block.is_heading:
                last_heading = block.content
            elif chunk_start in block.content:
                return last_heading
        return last_heading
