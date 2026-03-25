"""Load and manage the FinDocs evaluation dataset.

Provides utilities to load evaluation questions from JSON, sample subsets
(optionally filtered by question type), append new questions, and compute
dataset statistics.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EvalQuestion(BaseModel):
    """A single evaluation question with metadata for stratified sampling."""

    id: str = Field(description="Unique identifier for the question.")
    question: str = Field(description="The evaluation question text.")
    ground_truth_answer: str = Field(
        description="Expected ground-truth answer for scoring."
    )
    doc_type: str = Field(
        description="Document type the question targets "
        "(e.g. 'rbi_circular', 'sebi_factsheet')."
    )
    question_type: str = Field(
        description="Category of question "
        "(e.g. 'factoid', 'numerical', 'multi-hop', 'summarization')."
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Subjective difficulty rating."
    )


# ---------------------------------------------------------------------------
# Dataset manager
# ---------------------------------------------------------------------------


class EvalDatasetManager:
    """Manages the lifecycle of a JSON-backed evaluation dataset.

    The dataset file is expected to be a JSON array of objects, each
    conforming to the ``EvalQuestion`` schema.

    Attributes:
        _questions: In-memory store of loaded questions.
    """

    def __init__(self) -> None:
        """Initialise the manager with an empty question list."""
        self._questions: list[EvalQuestion] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Path) -> list[EvalQuestion]:
        """Load evaluation questions from a JSON file.

        The file must contain a JSON array of objects, each with the fields
        defined in ``EvalQuestion``.

        Args:
            path: Path to the evaluation dataset JSON file.

        Returns:
            A list of ``EvalQuestion`` instances.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            pydantic.ValidationError: If any record fails validation.
        """
        log = logger.bind(path=str(path))
        log.info("eval_dataset.load.start")

        raw_text = path.read_text(encoding="utf-8")
        raw_data = json.loads(raw_text)

        if not isinstance(raw_data, list):
            raise ValueError(
                f"Expected a JSON array at top level, got {type(raw_data).__name__}"
            )

        self._questions = [EvalQuestion.model_validate(item) for item in raw_data]
        log.info("eval_dataset.load.done", count=len(self._questions))
        return list(self._questions)

    def sample(
        self,
        n: int,
        question_types: list[str] | None = None,
    ) -> list[EvalQuestion]:
        """Return a random sample from the loaded questions.

        Optionally filters to only questions whose ``question_type`` is in
        the provided list before sampling.

        Args:
            n: Number of questions to sample.
            question_types: If provided, only questions matching one of
                these types are eligible for sampling.

        Returns:
            A list of up to ``n`` ``EvalQuestion`` instances.  If fewer
            questions are available than requested, all eligible questions
            are returned (shuffled).

        Raises:
            RuntimeError: If no questions have been loaded yet.
        """
        if not self._questions:
            raise RuntimeError(
                "No questions loaded. Call load() before sample()."
            )

        pool = self._questions
        if question_types is not None:
            allowed = set(question_types)
            pool = [q for q in pool if q.question_type in allowed]
            logger.debug(
                "eval_dataset.sample.filtered",
                question_types=question_types,
                pool_size=len(pool),
            )

        if n >= len(pool):
            logger.warning(
                "eval_dataset.sample.requested_exceeds_pool",
                requested=n,
                available=len(pool),
            )
            result = list(pool)
            random.shuffle(result)
            return result

        return random.sample(pool, n)

    def add_questions(
        self,
        questions: list[EvalQuestion],
        path: Path,
    ) -> None:
        """Append new questions to the dataset file and in-memory store.

        Existing questions in the file are preserved; duplicates (by ``id``)
        are silently skipped.

        Args:
            questions: New questions to add.
            path: Path to the evaluation dataset JSON file.  The file is
                created if it does not exist.
        """
        log = logger.bind(path=str(path), new_count=len(questions))
        log.info("eval_dataset.add_questions.start")

        # Load existing data if the file exists
        existing: list[dict[str, object]] = []
        if path.exists():
            raw_text = path.read_text(encoding="utf-8")
            existing = json.loads(raw_text)
            if not isinstance(existing, list):
                raise ValueError(
                    f"Expected a JSON array at top level, got {type(existing).__name__}"
                )

        existing_ids: set[str] = {
            str(item.get("id", "")) for item in existing
        }

        added = 0
        for q in questions:
            if q.id in existing_ids:
                log.debug("eval_dataset.add_questions.skip_duplicate", id=q.id)
                continue
            existing.append(q.model_dump())
            existing_ids.add(q.id)
            added += 1

        # Write the merged dataset back
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Refresh in-memory store
        self._questions = [EvalQuestion.model_validate(item) for item in existing]

        log.info("eval_dataset.add_questions.done", added=added, total=len(self._questions))

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Compute summary statistics over the loaded questions.

        Returns:
            A dictionary with three keys:

            - ``by_question_type``: counts keyed by ``question_type``.
            - ``by_doc_type``: counts keyed by ``doc_type``.
            - ``by_difficulty``: counts keyed by ``difficulty``.

        Raises:
            RuntimeError: If no questions have been loaded yet.
        """
        if not self._questions:
            raise RuntimeError(
                "No questions loaded. Call load() before get_stats()."
            )

        by_question_type: Counter[str] = Counter()
        by_doc_type: Counter[str] = Counter()
        by_difficulty: Counter[str] = Counter()

        for q in self._questions:
            by_question_type[q.question_type] += 1
            by_doc_type[q.doc_type] += 1
            by_difficulty[q.difficulty] += 1

        stats: dict[str, dict[str, int]] = {
            "by_question_type": dict(by_question_type),
            "by_doc_type": dict(by_doc_type),
            "by_difficulty": dict(by_difficulty),
        }

        logger.info(
            "eval_dataset.get_stats",
            total=len(self._questions),
            question_types=len(by_question_type),
            doc_types=len(by_doc_type),
        )
        return stats
