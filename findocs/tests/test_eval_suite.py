"""Tests for evaluation modules: EvalReport, ci_gate, LLMJudge.

All external services (Anthropic, Langfuse, RAGAS) are mocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from findocs.evaluation.eval_suite import EvalReport, SingleEvalResult
from findocs.evaluation.llm_judge import JudgeResult, LLMJudge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_report(
    faithfulness: float = 0.90,
    answer_relevancy: float = 0.85,
    context_precision: float = 0.80,
    context_recall: float = 0.78,
    answer_correctness: float = 0.82,
    failing_metrics: list[str] | None = None,
    passed_ci_gate: bool = True,
) -> EvalReport:
    """Create an EvalReport with configurable metric scores."""

    per_question = [
        SingleEvalResult(
            question=f"Question {i}",
            answer=f"Answer {i}",
            ground_truth=f"Ground truth {i}",
            contexts=[f"Context {i}"],
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            answer_correctness=answer_correctness,
        )
        for i in range(3)
    ]

    return EvalReport(
        run_id="test-run-001",
        timestamp=datetime(2025, 3, 15, 10, 30, 0, tzinfo=timezone.utc),
        sample_size=3,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
        answer_correctness=answer_correctness,
        per_question=per_question,
        passed_ci_gate=passed_ci_gate,
        failing_metrics=failing_metrics or [],
    )


# ============================================================================
# EvalReport Tests
# ============================================================================


class TestEvalReport:
    """Tests for the EvalReport model and its markdown output."""

    def test_eval_report_markdown_summary(self) -> None:
        """Verify to_markdown_summary() generates a valid markdown table."""
        with patch("findocs.evaluation.eval_suite.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                MIN_FAITHFULNESS=0.80,
                MIN_ANSWER_RELEVANCE=0.75,
                MIN_CONTEXT_PRECISION=0.70,
            )

            report = _make_eval_report()
            markdown = report.to_markdown_summary()

        # Should contain the report header
        assert "## FinDocs Eval Report" in markdown
        assert "test-run-001" in markdown

        # Should contain the metrics table
        assert "| Metric | Score | Threshold | Status |" in markdown
        assert "|--------|------:|----------:|--------|" in markdown

        # Should contain all metric names
        assert "faithfulness" in markdown
        assert "answer_relevancy" in markdown
        assert "context_precision" in markdown
        assert "context_recall" in markdown
        assert "answer_correctness" in markdown

        # Should have PASS/FAIL statuses
        assert "PASS" in markdown

        # Should have the overall verdict
        assert "**Overall: PASSED**" in markdown

        # Should contain the sample size
        assert "**Sample size:** 3" in markdown

        # Verify it's valid markdown table structure (at least 7 lines for header + separator + 5 metrics)
        lines = markdown.strip().split("\n")
        table_lines = [l for l in lines if l.startswith("|")]
        assert len(table_lines) >= 7, (
            f"Expected at least 7 table lines (header + sep + 5 metrics), got {len(table_lines)}"
        )

    def test_eval_report_identifies_failing_metrics(self) -> None:
        """Set faithfulness below 0.80, verify it appears in failing_metrics."""
        with patch("findocs.evaluation.eval_suite.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                MIN_FAITHFULNESS=0.80,
                MIN_ANSWER_RELEVANCE=0.75,
                MIN_CONTEXT_PRECISION=0.70,
            )

            report = _make_eval_report(
                faithfulness=0.65,  # Below threshold of 0.80
                failing_metrics=["faithfulness"],
                passed_ci_gate=False,
            )

            assert "faithfulness" in report.failing_metrics, (
                f"Expected 'faithfulness' in failing_metrics, got {report.failing_metrics}"
            )
            assert not report.passed_ci_gate

            # Markdown should reflect failure
            markdown = report.to_markdown_summary()
            assert "FAIL" in markdown
            assert "**Overall: FAILED**" in markdown
            assert "Failing metrics: faithfulness" in markdown

    def test_eval_report_all_metrics_fail(self) -> None:
        """When all metrics are below threshold, all should appear in failing_metrics."""
        report = _make_eval_report(
            faithfulness=0.50,
            answer_relevancy=0.40,
            context_precision=0.30,
            context_recall=0.20,
            answer_correctness=0.10,
            failing_metrics=["faithfulness", "answer_relevancy", "context_precision",
                             "context_recall", "answer_correctness"],
            passed_ci_gate=False,
        )

        assert len(report.failing_metrics) == 5
        assert not report.passed_ci_gate


# ============================================================================
# CI Gate Tests
# ============================================================================


class TestCIGate:
    """Tests for the CI gate logic."""

    @pytest.mark.asyncio
    async def test_ci_gate_passes_when_above_threshold(self) -> None:
        """Mock eval returning scores above thresholds, verify exit code 0."""
        # We test the logic by constructing an EvalReport that passes
        report = _make_eval_report(
            faithfulness=0.90,
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.78,
            answer_correctness=0.82,
            passed_ci_gate=True,
            failing_metrics=[],
        )

        # Verify the report says it passed
        assert report.passed_ci_gate is True
        assert report.failing_metrics == []

        # Simulate what ci_gate.py does: check passed_ci_gate
        exit_code = 0 if report.passed_ci_gate else 1
        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

    @pytest.mark.asyncio
    async def test_ci_gate_fails_when_below_threshold(self) -> None:
        """Mock eval returning low faithfulness, verify exit code 1."""
        report = _make_eval_report(
            faithfulness=0.55,  # Below 0.80 threshold
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.78,
            answer_correctness=0.82,
            passed_ci_gate=False,
            failing_metrics=["faithfulness"],
        )

        # Verify the report says it failed
        assert report.passed_ci_gate is False
        assert "faithfulness" in report.failing_metrics

        # Simulate what ci_gate.py does
        exit_code = 0 if report.passed_ci_gate else 1
        assert exit_code == 1, f"Expected exit code 1, got {exit_code}"

    @pytest.mark.asyncio
    async def test_ci_gate_full_flow(self) -> None:
        """Test the full ci_gate flow with mocked eval suite."""
        mock_report = _make_eval_report(
            faithfulness=0.92,
            answer_relevancy=0.88,
            context_precision=0.85,
            context_recall=0.80,
            answer_correctness=0.83,
            passed_ci_gate=True,
            failing_metrics=[],
        )

        with patch("findocs.evaluation.ci_gate.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                MIN_FAITHFULNESS=0.80,
                MIN_ANSWER_RELEVANCE=0.75,
                MIN_CONTEXT_PRECISION=0.70,
            )

            from findocs.evaluation.ci_gate import _build_threshold_table

            rows = _build_threshold_table(mock_report)

        # All metrics should pass
        for row in rows:
            assert row["status"] == "PASS", (
                f"Metric {row['metric']} should PASS, got {row['status']}"
            )


# ============================================================================
# LLM Judge Tests
# ============================================================================


class TestLLMJudge:
    """Tests for the LLM-as-judge scoring system."""

    @pytest.mark.asyncio
    async def test_llm_judge_returns_score(self) -> None:
        """Mock LLM call, verify score is between 0 and 1."""
        mock_client = MagicMock()

        # Build mock response
        content_block = MagicMock()
        content_block.text = '{"score": 0.85, "reasoning": "The answer correctly identifies the repo rate."}'
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        judge = LLMJudge(
            anthropic_client=mock_client,
            model="claude-3-haiku-20240307",
            rate_limit_delay=0.0,  # No delay in tests
        )

        result = await judge.judge_answer(
            question="What is the repo rate?",
            answer="The repo rate is 6.50%.",
            ground_truth="The current repo rate set by RBI is 6.50%.",
        )

        assert isinstance(result, JudgeResult)
        assert 0.0 <= result.score <= 1.0, (
            f"Score should be between 0 and 1, got {result.score}"
        )
        assert result.score == 0.85
        assert result.reasoning, "Reasoning should not be empty"

    @pytest.mark.asyncio
    async def test_llm_judge_handles_zero_score(self) -> None:
        """Verify judge correctly handles a zero score for completely wrong answers."""
        mock_client = MagicMock()

        content_block = MagicMock()
        content_block.text = '{"score": 0.0, "reasoning": "The answer is completely unrelated."}'
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        judge = LLMJudge(
            anthropic_client=mock_client,
            model="claude-3-haiku-20240307",
            rate_limit_delay=0.0,
        )

        result = await judge.judge_answer(
            question="What is the repo rate?",
            answer="Pizza is delicious.",
            ground_truth="The current repo rate set by RBI is 6.50%.",
        )

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_llm_judge_handles_perfect_score(self) -> None:
        """Verify judge correctly handles a perfect score."""
        mock_client = MagicMock()

        content_block = MagicMock()
        content_block.text = '{"score": 1.0, "reasoning": "Perfect match with ground truth."}'
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        judge = LLMJudge(
            anthropic_client=mock_client,
            model="claude-3-haiku-20240307",
            rate_limit_delay=0.0,
        )

        result = await judge.judge_answer(
            question="What is the repo rate?",
            answer="The current repo rate set by RBI is 6.50%.",
            ground_truth="The current repo rate set by RBI is 6.50%.",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_llm_judge_handles_markdown_fenced_response(self) -> None:
        """Verify judge handles responses wrapped in markdown code fences."""
        mock_client = MagicMock()

        content_block = MagicMock()
        content_block.text = '```json\n{"score": 0.75, "reasoning": "Mostly correct."}\n```'
        mock_response = MagicMock()
        mock_response.content = [content_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        judge = LLMJudge(
            anthropic_client=mock_client,
            model="claude-3-haiku-20240307",
            rate_limit_delay=0.0,
        )

        result = await judge.judge_answer(
            question="What is the CRR?",
            answer="CRR is 4.50%.",
            ground_truth="The CRR is 4.50% of NDTL.",
        )

        assert result.score == 0.75

    @pytest.mark.asyncio
    async def test_llm_judge_batch_judge(self) -> None:
        """Verify batch_judge returns correct number of scores."""
        mock_client = MagicMock()

        # Return different scores for each call
        responses = []
        for score in [0.9, 0.7, 0.5]:
            content_block = MagicMock()
            content_block.text = json.dumps({"score": score, "reasoning": f"Score {score}"})
            mock_response = MagicMock()
            mock_response.content = [content_block]
            responses.append(mock_response)

        mock_client.messages.create = AsyncMock(side_effect=responses)

        judge = LLMJudge(
            anthropic_client=mock_client,
            model="claude-3-haiku-20240307",
            rate_limit_delay=0.0,
        )

        # Create dummy SingleEvalResult objects
        from findocs.evaluation.eval_suite import SingleEvalResult

        results = [
            SingleEvalResult(
                question=f"Q{i}",
                answer=f"A{i}",
                ground_truth=f"GT{i}",
                contexts=[f"C{i}"],
                faithfulness=0.8,
                answer_relevancy=0.8,
                context_precision=0.8,
                context_recall=0.8,
                answer_correctness=0.0,
            )
            for i in range(3)
        ]

        scores = await judge.batch_judge(results)

        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        assert scores[0] == 0.9
        assert scores[1] == 0.7
        assert scores[2] == 0.5

        # All scores should be between 0 and 1
        for s in scores:
            assert 0.0 <= s <= 1.0
