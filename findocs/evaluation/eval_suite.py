"""RAGAS-based evaluation runner for the FinDocs RAG pipeline.

Runs a suite of five quality metrics against a sampled evaluation dataset,
produces a structured report, and logs every run to Langfuse for
longitudinal tracking.

# ARCHITECTURE DECISION: Why eval runs in CI
# Shift-left quality: catch regressions before they reach users. By running a 50-question
# eval sample on every PR that touches serving/retrieval/evaluation code, we detect
# prompt regressions, retrieval quality drops, and model degradation at PR review time —
# not after deployment. The eval gate blocks merge if any metric drops below threshold,
# making quality a hard constraint rather than a soft aspiration.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog
from datasets import Dataset
from langfuse import Langfuse
from pydantic import BaseModel, Field
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from findocs.config.config import get_settings
from findocs.evaluation.llm_judge import LLMJudge

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol that any RAG pipeline must satisfy to be evaluable
# ---------------------------------------------------------------------------


@runtime_checkable
class RAGPipeline(Protocol):
    """Minimal interface a RAG pipeline must expose for evaluation."""

    async def aquery(self, question: str) -> dict[str, Any]:
        """Run a RAG query and return ``{"answer": str, "contexts": list[str]}``."""
        ...


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SingleEvalResult(BaseModel):
    """Result of evaluating a single question-answer pair."""

    question: str = Field(description="The evaluation question that was asked.")
    answer: str = Field(description="The answer produced by the RAG pipeline.")
    ground_truth: str = Field(description="The expected ground-truth answer.")
    contexts: list[str] = Field(
        description="Retrieved context passages used to generate the answer."
    )
    faithfulness: float = Field(
        description="RAGAS faithfulness score (0-1).",
        ge=0.0,
        le=1.0,
    )
    answer_relevancy: float = Field(
        description="RAGAS answer relevancy score (0-1).",
        ge=0.0,
        le=1.0,
    )
    context_precision: float = Field(
        description="RAGAS context precision score (0-1).",
        ge=0.0,
        le=1.0,
    )
    context_recall: float = Field(
        description="RAGAS context recall score (0-1).",
        ge=0.0,
        le=1.0,
    )
    answer_correctness: float = Field(
        description="LLM-judge answer correctness score (0-1).",
        ge=0.0,
        le=1.0,
    )


class EvalReport(BaseModel):
    """Aggregate evaluation report over an entire eval run."""

    run_id: str = Field(description="Unique identifier for this evaluation run.")
    timestamp: datetime = Field(
        description="UTC timestamp when the evaluation completed."
    )
    sample_size: int = Field(description="Number of questions evaluated.")
    faithfulness: float = Field(
        description="Mean faithfulness across all questions.",
        ge=0.0,
        le=1.0,
    )
    answer_relevancy: float = Field(
        description="Mean answer relevancy across all questions.",
        ge=0.0,
        le=1.0,
    )
    context_precision: float = Field(
        description="Mean context precision across all questions.",
        ge=0.0,
        le=1.0,
    )
    context_recall: float = Field(
        description="Mean context recall across all questions.",
        ge=0.0,
        le=1.0,
    )
    answer_correctness: float = Field(
        description="Mean LLM-judge answer correctness across all questions.",
        ge=0.0,
        le=1.0,
    )
    per_question: list[SingleEvalResult] = Field(
        description="Per-question detailed results."
    )
    passed_ci_gate: bool = Field(
        description="Whether all metrics met their CI threshold."
    )
    failing_metrics: list[str] = Field(
        default_factory=list,
        description="Names of metrics that fell below their CI threshold.",
    )

    def to_markdown_summary(self) -> str:
        """Generate a Markdown table suitable for a GitHub PR comment.

        Returns:
            A Markdown-formatted string containing a summary table with
            each metric, its value, the CI threshold, and pass/fail status,
            followed by an overall verdict.
        """
        settings = get_settings()
        thresholds: dict[str, float] = {
            "faithfulness": settings.MIN_FAITHFULNESS,
            "answer_relevancy": settings.MIN_ANSWER_RELEVANCE,
            "context_precision": settings.MIN_CONTEXT_PRECISION,
            "context_recall": 0.70,
            "answer_correctness": 0.70,
        }

        scores: dict[str, float] = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_correctness": self.answer_correctness,
        }

        lines: list[str] = [
            f"## FinDocs Eval Report — `{self.run_id}`",
            "",
            f"**Sample size:** {self.sample_size} | "
            f"**Timestamp:** {self.timestamp.isoformat()}",
            "",
            "| Metric | Score | Threshold | Status |",
            "|--------|------:|----------:|--------|",
        ]

        for metric_name, score in scores.items():
            threshold = thresholds.get(metric_name, 0.70)
            status = "PASS" if score >= threshold else "FAIL"
            lines.append(
                f"| {metric_name} | {score:.4f} | {threshold:.2f} | {status} |"
            )

        overall = "PASSED" if self.passed_ci_gate else "FAILED"
        lines.append("")
        lines.append(f"**Overall: {overall}**")

        if self.failing_metrics:
            lines.append("")
            lines.append(
                f"Failing metrics: {', '.join(self.failing_metrics)}"
            )

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Persist the report as a JSON file.

        Args:
            path: Filesystem path where the JSON report will be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        logger.info("eval_report.saved", path=str(path))


# ---------------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------------


class EvalSuite:
    """Orchestrates end-to-end RAG evaluation using RAGAS metrics and a
    custom LLM judge for answer correctness.

    Attributes:
        _settings: Application settings (thresholds, API keys, etc.).
        _llm_judge: LLM-as-judge instance for answer correctness scoring.
        _langfuse: Langfuse client for observability logging.
    """

    def __init__(self) -> None:
        """Initialise the evaluation suite with settings, LLM judge, and Langfuse."""
        self._settings = get_settings()
        self._llm_judge = LLMJudge.from_settings(self._settings)
        self._langfuse = Langfuse(
            public_key=self._settings.LANGFUSE_PUBLIC_KEY,
            secret_key=self._settings.LANGFUSE_SECRET_KEY,
            host=self._settings.LANGFUSE_HOST,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        rag_pipeline: RAGPipeline,
        eval_dataset_path: str | Path,
        sample_size: int | None = None,
        run_id: str | None = None,
    ) -> EvalReport:
        """Execute a full evaluation run.

        For each question in the (optionally sampled) dataset, the RAG
        pipeline is queried, RAGAS metrics are computed in batch, and the
        custom LLM judge scores answer correctness.  Results are logged to
        Langfuse and returned as a structured ``EvalReport``.

        Args:
            rag_pipeline: An object satisfying the ``RAGPipeline`` protocol.
            eval_dataset_path: Path to the evaluation dataset JSON file.
            sample_size: Number of questions to evaluate.  ``None`` means
                evaluate the entire dataset.
            run_id: Optional identifier for this run.  Generated
                automatically when omitted.

        Returns:
            A complete ``EvalReport`` with per-question and aggregate scores.
        """
        from findocs.evaluation.eval_dataset import EvalDatasetManager

        run_id = run_id or uuid.uuid4().hex[:12]
        log = logger.bind(run_id=run_id)
        log.info("eval_suite.run.start", dataset=str(eval_dataset_path))

        # Load and optionally sample the evaluation dataset
        dataset_mgr = EvalDatasetManager()
        questions = dataset_mgr.load(Path(eval_dataset_path))
        if sample_size is not None and sample_size < len(questions):
            questions = dataset_mgr.sample(sample_size)
        log.info("eval_suite.run.dataset_loaded", total_questions=len(questions))

        # Create a Langfuse trace for the whole run
        trace = self._langfuse.trace(
            name="eval_run",
            id=run_id,
            metadata={"sample_size": len(questions)},
        )

        # --- Step 1: Run every question through the RAG pipeline ----------
        rag_outputs: list[dict[str, Any]] = []
        for idx, q in enumerate(questions):
            log.debug("eval_suite.run.query", idx=idx, question=q.question[:80])
            span = trace.span(
                name="rag_query",
                input={"question": q.question},
            )
            try:
                output = await rag_pipeline.aquery(q.question)
            except Exception:
                log.exception("eval_suite.run.query_failed", question=q.question[:80])
                output = {"answer": "", "contexts": []}
            span.end(output=output)
            rag_outputs.append(output)

        # --- Step 2: Build a HuggingFace Dataset for RAGAS -----------------
        ragas_data = {
            "question": [q.question for q in questions],
            "answer": [o.get("answer", "") for o in rag_outputs],
            "contexts": [o.get("contexts", []) for o in rag_outputs],
            "ground_truth": [q.ground_truth_answer for q in questions],
        }
        hf_dataset = Dataset.from_dict(ragas_data)

        # --- Step 3: Run RAGAS evaluate (synchronous) ----------------------
        log.info("eval_suite.run.ragas_start")
        ragas_result = await asyncio.to_thread(
            evaluate,
            dataset=hf_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        ragas_df = ragas_result.to_pandas()
        log.info("eval_suite.run.ragas_done")

        # --- Step 4: LLM-judge answer correctness --------------------------
        log.info("eval_suite.run.llm_judge_start")
        preliminary_results: list[dict[str, Any]] = []
        for idx in range(len(questions)):
            preliminary_results.append(
                {
                    "question": questions[idx].question,
                    "answer": rag_outputs[idx].get("answer", ""),
                    "ground_truth": questions[idx].ground_truth_answer,
                    "contexts": rag_outputs[idx].get("contexts", []),
                    "faithfulness": float(ragas_df.iloc[idx].get("faithfulness", 0.0)),
                    "answer_relevancy": float(
                        ragas_df.iloc[idx].get("answer_relevancy", 0.0)
                    ),
                    "context_precision": float(
                        ragas_df.iloc[idx].get("context_precision", 0.0)
                    ),
                    "context_recall": float(
                        ragas_df.iloc[idx].get("context_recall", 0.0)
                    ),
                }
            )

        # Build temporary SingleEvalResult objects with placeholder correctness
        # so we can pass them to batch_judge
        temp_results = [
            SingleEvalResult(**{**pr, "answer_correctness": 0.0})
            for pr in preliminary_results
        ]
        correctness_scores = await self._llm_judge.batch_judge(temp_results)
        log.info("eval_suite.run.llm_judge_done")

        # --- Step 5: Assemble per-question results -------------------------
        per_question: list[SingleEvalResult] = []
        for idx, pr in enumerate(preliminary_results):
            result = SingleEvalResult(
                **{**pr, "answer_correctness": correctness_scores[idx]}
            )
            per_question.append(result)

        # --- Step 6: Compute aggregate means --------------------------------
        n = len(per_question)
        mean_faithfulness = sum(r.faithfulness for r in per_question) / n
        mean_answer_relevancy = sum(r.answer_relevancy for r in per_question) / n
        mean_context_precision = sum(r.context_precision for r in per_question) / n
        mean_context_recall = sum(r.context_recall for r in per_question) / n
        mean_answer_correctness = sum(r.answer_correctness for r in per_question) / n

        # --- Step 7: Check CI thresholds ------------------------------------
        threshold_map: dict[str, tuple[float, float]] = {
            "faithfulness": (mean_faithfulness, self._settings.MIN_FAITHFULNESS),
            "answer_relevancy": (
                mean_answer_relevancy,
                self._settings.MIN_ANSWER_RELEVANCE,
            ),
            "context_precision": (
                mean_context_precision,
                self._settings.MIN_CONTEXT_PRECISION,
            ),
            "context_recall": (mean_context_recall, 0.70),
            "answer_correctness": (mean_answer_correctness, 0.70),
        }

        failing_metrics: list[str] = [
            name
            for name, (score, threshold) in threshold_map.items()
            if score < threshold
        ]
        passed_ci_gate = len(failing_metrics) == 0

        report = EvalReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            sample_size=n,
            faithfulness=mean_faithfulness,
            answer_relevancy=mean_answer_relevancy,
            context_precision=mean_context_precision,
            context_recall=mean_context_recall,
            answer_correctness=mean_answer_correctness,
            per_question=per_question,
            passed_ci_gate=passed_ci_gate,
            failing_metrics=failing_metrics,
        )

        # --- Step 8: Log to Langfuse ----------------------------------------
        trace.update(
            output={
                "passed_ci_gate": passed_ci_gate,
                "faithfulness": mean_faithfulness,
                "answer_relevancy": mean_answer_relevancy,
                "context_precision": mean_context_precision,
                "context_recall": mean_context_recall,
                "answer_correctness": mean_answer_correctness,
                "failing_metrics": failing_metrics,
            },
        )
        self._langfuse.flush()
        log.info(
            "eval_suite.run.complete",
            passed=passed_ci_gate,
            failing_metrics=failing_metrics,
        )

        return report

    async def run_single(
        self,
        question: str,
        ground_truth: str,
        rag_pipeline: RAGPipeline,
    ) -> SingleEvalResult:
        """Evaluate a single question-answer pair.

        Useful for interactive debugging or ad-hoc spot checks.

        Args:
            question: The evaluation question.
            ground_truth: The expected ground-truth answer.
            rag_pipeline: An object satisfying the ``RAGPipeline`` protocol.

        Returns:
            A ``SingleEvalResult`` with all five metric scores.
        """
        log = logger.bind(question=question[:80])
        log.info("eval_suite.run_single.start")

        # Query the pipeline
        output = await rag_pipeline.aquery(question)
        answer: str = output.get("answer", "")
        contexts: list[str] = output.get("contexts", [])

        # RAGAS metrics via a single-row dataset
        hf_dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
        )
        ragas_result = await asyncio.to_thread(
            evaluate,
            dataset=hf_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        ragas_df = ragas_result.to_pandas()
        row = ragas_df.iloc[0]

        # LLM-judge answer correctness
        judge_result = await self._llm_judge.judge_answer(question, answer, ground_truth)

        result = SingleEvalResult(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
            faithfulness=float(row.get("faithfulness", 0.0)),
            answer_relevancy=float(row.get("answer_relevancy", 0.0)),
            context_precision=float(row.get("context_precision", 0.0)),
            context_recall=float(row.get("context_recall", 0.0)),
            answer_correctness=judge_result.score,
        )
        log.info(
            "eval_suite.run_single.done",
            faithfulness=result.faithfulness,
            answer_correctness=result.answer_correctness,
        )
        return result
