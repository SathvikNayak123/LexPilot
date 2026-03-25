#!/usr/bin/env python3
"""CI gate for FinDocs RAG evaluation quality.

CLI entry point intended to run as ``python -m evaluation.ci_gate``.
Executes the evaluation suite against a configurable sample of the eval
dataset, checks every metric against its threshold, and exits with code 0
when all metrics pass or code 1 when any metric is below threshold.

Optionally writes a Markdown comment file suitable for posting to a GitHub
PR via CI automation.

# ARCHITECTURE DECISION: Why eval runs in CI
# Shift-left quality: catch regressions before they reach users. By running a 50-question
# eval sample on every PR that touches serving/retrieval/evaluation code, we detect
# prompt regressions, retrieval quality drops, and model degradation at PR review time —
# not after deployment. The eval gate blocks merge if any metric drops below threshold,
# making quality a hard constraint rather than a soft aspiration.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
import structlog

from findocs.config.config import get_settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_SAMPLE_SIZE: int = 50
_DEFAULT_REPORT_PATH: str = "eval_report.json"
_DEFAULT_EVAL_DATASET: str = "data/eval/eval_dataset.json"


def _build_threshold_table(report: Any) -> list[dict[str, str | float]]:
    """Build a list of metric rows for display.

    Each row contains the metric name, its score, the CI threshold, and
    whether it passed.

    Args:
        report: An ``EvalReport`` instance from ``eval_suite``.

    Returns:
        A list of dicts, one per metric, with keys ``metric``, ``score``,
        ``threshold``, and ``status``.
    """
    settings = get_settings()

    metrics: list[dict[str, str | float]] = [
        {
            "metric": "faithfulness",
            "score": report.faithfulness,
            "threshold": settings.MIN_FAITHFULNESS,
        },
        {
            "metric": "answer_relevancy",
            "score": report.answer_relevancy,
            "threshold": settings.MIN_ANSWER_RELEVANCE,
        },
        {
            "metric": "context_precision",
            "score": report.context_precision,
            "threshold": settings.MIN_CONTEXT_PRECISION,
        },
        {
            "metric": "context_recall",
            "score": report.context_recall,
            "threshold": 0.70,
        },
        {
            "metric": "answer_correctness",
            "score": report.answer_correctness,
            "threshold": 0.70,
        },
    ]

    for row in metrics:
        score = float(row["score"])
        threshold = float(row["threshold"])
        row["status"] = "PASS" if score >= threshold else "FAIL"

    return metrics


def _print_results_table(rows: list[dict[str, str | float]], passed: bool) -> None:
    """Print a formatted results table to stdout.

    Args:
        rows: Metric rows as returned by ``_build_threshold_table``.
        passed: Whether the overall gate passed.
    """
    header = f"{'Metric':<25} {'Score':>8} {'Threshold':>10} {'Status':>8}"
    separator = "-" * len(header)

    click.echo("")
    click.echo("FinDocs Evaluation CI Gate")
    click.echo(separator)
    click.echo(header)
    click.echo(separator)

    for row in rows:
        status_str = str(row["status"])
        score_str = f"{float(row['score']):.4f}"
        threshold_str = f"{float(row['threshold']):.2f}"
        metric_str = str(row["metric"])

        if status_str == "FAIL":
            line = click.style(
                f"{metric_str:<25} {score_str:>8} {threshold_str:>10} {status_str:>8}",
                fg="red",
            )
        else:
            line = click.style(
                f"{metric_str:<25} {score_str:>8} {threshold_str:>10} {status_str:>8}",
                fg="green",
            )
        click.echo(line)

    click.echo(separator)

    if passed:
        click.echo(click.style("OVERALL: PASSED", fg="green", bold=True))
    else:
        click.echo(click.style("OVERALL: FAILED", fg="red", bold=True))

    click.echo("")


async def _run_eval(
    sample_size: int,
    report_path: Path,
    comment_file: Path | None,
    dataset_path: Path,
) -> bool:
    """Execute the evaluation suite and process results.

    Args:
        sample_size: Number of questions to sample from the eval dataset.
        report_path: Path to write the JSON evaluation report.
        comment_file: Optional path to write a Markdown comment file for
            GitHub PR comments.
        dataset_path: Path to the evaluation dataset JSON file.

    Returns:
        ``True`` if all metrics passed their thresholds, ``False`` otherwise.
    """
    from findocs.evaluation.eval_suite import EvalSuite
    from findocs.serving.rag_pipeline import RAGPipelineService  # type: ignore[import-untyped]

    log = logger.bind(sample_size=sample_size, report_path=str(report_path))
    log.info("ci_gate.start")

    # Instantiate the RAG pipeline and evaluation suite
    pipeline = RAGPipelineService()
    suite = EvalSuite()

    # Run evaluation
    report = await suite.run(
        rag_pipeline=pipeline,
        eval_dataset_path=dataset_path,
        sample_size=sample_size,
    )

    # Save JSON report
    report.save(report_path)
    log.info("ci_gate.report_saved", path=str(report_path))

    # Print formatted table
    rows = _build_threshold_table(report)
    _print_results_table(rows, report.passed_ci_gate)

    # Write Markdown comment file if requested
    if comment_file is not None:
        markdown = report.to_markdown_summary()
        comment_file.parent.mkdir(parents=True, exist_ok=True)
        comment_file.write_text(markdown, encoding="utf-8")
        log.info("ci_gate.comment_file_written", path=str(comment_file))
        click.echo(f"Markdown comment written to: {comment_file}")

    # Log failing metrics
    if report.failing_metrics:
        log.warning(
            "ci_gate.failing_metrics",
            failing=report.failing_metrics,
        )
        click.echo(
            f"Failing metrics: {', '.join(report.failing_metrics)}",
            err=True,
        )

    return report.passed_ci_gate


@click.command()
@click.option(
    "--sample-size",
    type=int,
    default=_DEFAULT_SAMPLE_SIZE,
    show_default=True,
    help="Number of evaluation questions to sample.",
)
@click.option(
    "--report-path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_REPORT_PATH,
    show_default=True,
    help="Path to write the JSON evaluation report.",
)
@click.option(
    "--comment-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to write a Markdown summary for GitHub PR comments.",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, path_type=Path),
    default=_DEFAULT_EVAL_DATASET,
    show_default=True,
    help="Path to the evaluation dataset JSON file.",
)
def main(
    sample_size: int,
    report_path: Path,
    comment_file: Path | None,
    dataset_path: Path,
) -> None:
    """FinDocs Evaluation CI Gate.

    Runs the evaluation suite against a sample of the eval dataset and
    exits with code 0 if all metrics pass their thresholds, or code 1
    if any metric fails.
    """
    passed = asyncio.run(
        _run_eval(
            sample_size=sample_size,
            report_path=report_path,
            comment_file=comment_file,
            dataset_path=dataset_path,
        )
    )

    if passed:
        click.echo("CI gate passed. All metrics above threshold.")
        sys.exit(0)
    else:
        click.echo("CI gate FAILED. One or more metrics below threshold.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
