"""Synthetic QA dataset generation from a financial document corpus.

Uses GPT-4o as a teacher model to produce high-quality question–answer pairs
from chunked financial text.  The generator validates, deduplicates, and
splits the output into Alpaca-format JSONL files for SFTTrainer consumption.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from findocs.processing.chunker import Chunk

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

QuestionType = Literal["factual", "reasoning", "comparative", "procedural"]
DocType = Literal["rbi_circular", "sebi_factsheet", "nse_annual_report", "other"]

_QUESTION_TYPES: list[QuestionType] = ["factual", "reasoning", "comparative", "procedural"]
_DOC_TYPES: list[DocType] = ["rbi_circular", "sebi_factsheet", "nse_annual_report", "other"]

_SYSTEM_PROMPT = (
    "You are a senior Indian financial analyst and exam-question writer. "
    "Given a chunk of text from an Indian regulatory / financial document, "
    "generate exactly 3 question–answer pairs.\n\n"
    "Requirements:\n"
    "1. Every question MUST be answerable using ONLY the provided text.\n"
    "2. Include a mix of question types: factual recall AND reasoning / inference.\n"
    "3. Use professional financial terminology appropriate for Indian markets "
    "   (e.g. SEBI, RBI, NAV, AUM, repo rate, NPA).\n"
    "4. Answers must be self-contained, detailed, and cite specific numbers or "
    "   dates from the text when available.\n\n"
    "Return a JSON array of exactly 3 objects with keys:\n"
    '  "question": string,\n'
    '  "answer": string,\n'
    '  "question_type": one of ["factual", "reasoning", "comparative", "procedural"]\n'
)


class QAPair(BaseModel):
    """A single validated question–answer pair generated from a source chunk."""

    question: str = Field(..., min_length=11, description="The generated question.")
    answer: str = Field(..., min_length=21, description="The ground-truth answer.")
    question_type: QuestionType = Field(..., description="Taxonomy tag for the question.")
    source_doc_type: DocType = Field(
        default="other",
        description="Document type the source chunk belongs to.",
    )


class QADatasetGenerator:
    """Generates synthetic QA datasets from chunked financial documents.

    Uses the OpenAI ``gpt-4o`` model as a teacher to create diverse, high-quality
    QA pairs.  Provides utilities for deduplication (via embedding cosine
    similarity), train/val splitting, and export in Alpaca JSONL format.

    Args:
        openai_client: An initialised ``AsyncOpenAI`` client.
        teacher_model: The OpenAI model ID used for generation.
        embedding_model_name: SentenceTransformer model for deduplication.
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        teacher_model: str = "gpt-4o",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        self._client = openai_client
        self._teacher_model = teacher_model
        self._embedding_model_name = embedding_model_name
        self._embed_model: SentenceTransformer | None = None
        logger.info(
            "dataset_generator.init",
            teacher_model=teacher_model,
            embedding_model=embedding_model_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embed_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model so import time stays fast."""
        if self._embed_model is None:
            logger.info("dataset_generator.loading_embedding_model", model=self._embedding_model_name)
            self._embed_model = SentenceTransformer(self._embedding_model_name)
        return self._embed_model

    @staticmethod
    def _validate_pair(question: str, answer: str) -> bool:
        """Return True when a QA pair passes basic quality gates.

        Checks:
        * Question has more than 10 characters.
        * Answer has more than 20 characters.
        * The answer is not a substring of the question (prevents trivial
          echo answers).

        Args:
            question: The candidate question string.
            answer: The candidate answer string.

        Returns:
            ``True`` if the pair passes all checks, ``False`` otherwise.
        """
        if len(question.strip()) <= 10:
            return False
        if len(answer.strip()) <= 20:
            return False
        if answer.strip().lower() in question.strip().lower():
            return False
        return True

    @staticmethod
    def _infer_doc_type(chunk: Chunk) -> DocType:
        """Best-effort doc-type inference from chunk metadata.

        Args:
            chunk: The source ``Chunk`` object.

        Returns:
            A ``DocType`` literal value.
        """
        meta_type = (chunk.metadata or {}).get("doc_type", "")
        meta_type_lower = str(meta_type).lower()
        for dt in ("rbi_circular", "sebi_factsheet", "nse_annual_report"):
            if dt in meta_type_lower:
                return dt  # type: ignore[return-value]
        return "other"

    def _deduplicate_by_embedding(
        self,
        pairs: list[QAPair],
        threshold: float = 0.85,
    ) -> list[QAPair]:
        """Remove near-duplicate QA pairs using cosine similarity.

        For each pair, we embed ``question + " " + answer`` and drop any pair
        whose similarity to an already-accepted pair exceeds *threshold*.

        Args:
            pairs: Full list of candidate QA pairs.
            threshold: Cosine similarity ceiling; pairs above this are duplicates.

        Returns:
            Deduplicated list of ``QAPair`` objects.
        """
        if not pairs:
            return pairs

        model = self._get_embed_model()
        texts = [f"{p.question} {p.answer}" for p in pairs]
        embeddings: np.ndarray = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        keep_indices: list[int] = []
        kept_embeddings: list[np.ndarray] = []

        for idx, emb in enumerate(embeddings):
            is_duplicate = False
            for kept_emb in kept_embeddings:
                similarity = float(np.dot(emb, kept_emb))
                if similarity > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(idx)
                kept_embeddings.append(emb)

        deduped = [pairs[i] for i in keep_indices]
        removed = len(pairs) - len(deduped)
        if removed:
            logger.info(
                "dataset_generator.deduplication",
                original=len(pairs),
                kept=len(deduped),
                removed=removed,
                threshold=threshold,
            )
        return deduped

    @staticmethod
    def _to_alpaca(pair: QAPair) -> dict[str, str]:
        """Convert a QA pair to Alpaca instruction format.

        Args:
            pair: A validated ``QAPair``.

        Returns:
            Dict with keys ``instruction``, ``input``, ``output``.
        """
        return {
            "instruction": pair.question,
            "input": "",
            "output": pair.answer,
        }

    @staticmethod
    def _save_jsonl(records: list[dict], path: Path) -> None:
        """Write a list of dicts to a JSONL file.

        Args:
            records: Serialisable dicts, one per line.
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("dataset_generator.saved_jsonl", path=str(path), count=len(records))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_from_chunk(self, chunk: Chunk) -> list[QAPair]:
        """Prompt GPT-4o to generate 3 QA pairs from a single chunk.

        The teacher model is asked to produce exactly three question–answer
        objects grounded in the chunk text.  Each result is validated against
        minimum length requirements and the echo check.

        Args:
            chunk: A ``Chunk`` object containing the text to generate from.

        Returns:
            A list of validated ``QAPair`` objects (0-3 items).
        """
        if not chunk.text or len(chunk.text.strip()) < 50:
            logger.debug("dataset_generator.skip_short_chunk", chunk_id=chunk.chunk_id)
            return []

        user_content = (
            f"Source document type: {self._infer_doc_type(chunk)}\n\n"
            f"--- CHUNK START ---\n{chunk.text}\n--- CHUNK END ---"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._teacher_model,
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception:
            logger.exception("dataset_generator.openai_call_failed", chunk_id=chunk.chunk_id)
            return []

        raw_text = (response.choices[0].message.content or "").strip()

        # Parse the JSON response — it may be a bare array or {"pairs": [...]}.
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("dataset_generator.json_parse_failed", chunk_id=chunk.chunk_id)
            return []

        items: list[dict] = []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            # Accept any key that holds a list of dicts
            for val in parsed.values():
                if isinstance(val, list):
                    items = val
                    break

        doc_type = self._infer_doc_type(chunk)
        pairs: list[QAPair] = []

        for item in items:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            q_type_raw = str(item.get("question_type", "factual")).strip().lower()
            q_type: QuestionType = q_type_raw if q_type_raw in _QUESTION_TYPES else "factual"  # type: ignore[assignment]

            if not self._validate_pair(question, answer):
                logger.debug(
                    "dataset_generator.pair_failed_validation",
                    question_len=len(question),
                    answer_len=len(answer),
                )
                continue

            pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    question_type=q_type,
                    source_doc_type=doc_type,
                )
            )

        logger.debug(
            "dataset_generator.pairs_from_chunk",
            chunk_id=chunk.chunk_id,
            generated=len(pairs),
        )
        return pairs

    async def generate_dataset(
        self,
        chunks: list[Chunk],
        target_size: int = 3000,
        save_path: str | Path = "./data/train",
    ) -> int:
        """Generate a full training dataset from randomly sampled chunks.

        Workflow:
        1. Randomly sample enough chunks to reach *target_size* pairs
           (assuming ~3 pairs per chunk on average, with headroom).
        2. Call ``generate_from_chunk`` concurrently in batches.
        3. Deduplicate by embedding similarity > 0.85.
        4. Split 90/10 into train / validation JSONL files.

        Args:
            chunks: All available ``Chunk`` objects in the corpus.
            target_size: Desired number of QA pairs in the final dataset.
            save_path: Directory to write ``train.jsonl`` and ``val.jsonl``.

        Returns:
            Total number of QA pairs (train + val) written to disk.
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Over-sample to compensate for validation failures & dedup losses
        needed_chunks = min(len(chunks), int(target_size / 3 * 1.5) + 10)
        sampled = random.sample(chunks, needed_chunks) if len(chunks) > needed_chunks else list(chunks)
        random.shuffle(sampled)

        logger.info(
            "dataset_generator.generate_dataset_start",
            corpus_size=len(chunks),
            sampled=len(sampled),
            target_size=target_size,
        )

        all_pairs: list[QAPair] = []
        batch_size = 10  # concurrent API calls per batch
        for batch_start in range(0, len(sampled), batch_size):
            batch = sampled[batch_start : batch_start + batch_size]
            tasks = [self.generate_from_chunk(c) for c in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if isinstance(res, list):
                    all_pairs.extend(res)

            logger.info(
                "dataset_generator.batch_complete",
                batch_start=batch_start,
                pairs_so_far=len(all_pairs),
            )

            # Stop early if we have enough before dedup
            if len(all_pairs) >= int(target_size * 1.3):
                break

        # Deduplicate
        all_pairs = self._deduplicate_by_embedding(all_pairs, threshold=0.85)

        # Trim to target size
        if len(all_pairs) > target_size:
            all_pairs = all_pairs[:target_size]

        # 90/10 split
        split_idx = int(len(all_pairs) * 0.9)
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]

        train_records = [self._to_alpaca(p) for p in train_pairs]
        val_records = [self._to_alpaca(p) for p in val_pairs]

        self._save_jsonl(train_records, save_dir / "train.jsonl")
        self._save_jsonl(val_records, save_dir / "val.jsonl")

        total = len(train_records) + len(val_records)
        logger.info(
            "dataset_generator.generate_dataset_done",
            total=total,
            train=len(train_records),
            val=len(val_records),
        )
        return total

    async def generate_eval_dataset(
        self,
        chunks: list[Chunk],
        n: int = 200,
        save_path: str | Path = "./data/eval/eval_questions.jsonl",
    ) -> None:
        """Generate a high-quality evaluation dataset with diverse coverage.

        Each evaluation record includes the ground-truth answer plus metadata
        so downstream evaluation (RAGAS, faithfulness, etc.) can stratify by
        document type and question type.

        Sampling strategy: chunks are bucketed by inferred ``doc_type`` and
        questions are drawn from each bucket proportionally so that every
        document type is represented.

        Args:
            chunks: All available ``Chunk`` objects in the corpus.
            n: Number of evaluation QA pairs to produce.
            save_path: Output file path (JSONL).
        """
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        # Bucket chunks by doc type for diverse sampling
        buckets: dict[DocType, list[Chunk]] = {}
        for chunk in chunks:
            dt = self._infer_doc_type(chunk)
            buckets.setdefault(dt, []).append(chunk)

        # Proportional allocation across doc types
        num_types = max(len(buckets), 1)
        per_type = max(n // num_types, 1)

        selected_chunks: list[Chunk] = []
        for dt, dt_chunks in buckets.items():
            # Each chunk yields ~3 pairs, so we need per_type/3 chunks per type
            needed = min(len(dt_chunks), max(per_type // 3 + 2, 1))
            selected_chunks.extend(random.sample(dt_chunks, needed))

        random.shuffle(selected_chunks)

        logger.info(
            "dataset_generator.eval_generation_start",
            target_n=n,
            selected_chunks=len(selected_chunks),
            doc_type_buckets={k: len(v) for k, v in buckets.items()},
        )

        all_pairs: list[QAPair] = []
        batch_size = 10
        for batch_start in range(0, len(selected_chunks), batch_size):
            batch = selected_chunks[batch_start : batch_start + batch_size]
            tasks = [self.generate_from_chunk(c) for c in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    all_pairs.extend(res)
            if len(all_pairs) >= int(n * 1.2):
                break

        # Deduplicate eval set more aggressively
        all_pairs = self._deduplicate_by_embedding(all_pairs, threshold=0.80)

        # Ensure diversity: try to balance question types
        type_buckets: dict[QuestionType, list[QAPair]] = {}
        for pair in all_pairs:
            type_buckets.setdefault(pair.question_type, []).append(pair)

        balanced: list[QAPair] = []
        per_qtype = max(n // len(_QUESTION_TYPES), 1)
        for qt in _QUESTION_TYPES:
            candidates = type_buckets.get(qt, [])
            balanced.extend(candidates[:per_qtype])

        # Fill remaining slots from any type
        seen_questions = {p.question for p in balanced}
        for pair in all_pairs:
            if len(balanced) >= n:
                break
            if pair.question not in seen_questions:
                balanced.append(pair)
                seen_questions.add(pair.question)

        balanced = balanced[:n]

        # Write eval JSONL
        eval_records: list[dict[str, str]] = []
        for pair in balanced:
            # Compute a deterministic source_doc hash as a stand-in for the
            # actual document path (not available at pair level).
            source_hash = hashlib.sha256(pair.question.encode()).hexdigest()[:12]
            eval_records.append(
                {
                    "question": pair.question,
                    "ground_truth_answer": pair.answer,
                    "source_doc": f"chunk_{source_hash}",
                    "doc_type": pair.source_doc_type,
                    "question_type": pair.question_type,
                }
            )

        self._save_jsonl(eval_records, save_file)
        logger.info(
            "dataset_generator.eval_dataset_done",
            total=len(eval_records),
            question_types={qt: sum(1 for r in eval_records if r["question_type"] == qt) for qt in _QUESTION_TYPES},
        )
