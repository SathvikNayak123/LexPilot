"""Quality-drift detection via weekly batch evaluation.

Compares current RAG evaluation scores against a stored baseline in Postgres.
When any metric drops beyond a configurable threshold, a Slack alert is fired.
If faithfulness specifically drops more than 10 %, an automatic retrain is
triggered by writing a flag to the ``retrain_triggers`` table.
"""

from __future__ import annotations

# ARCHITECTURE DECISION: Why weekly eval over continuous
# Running eval on every production query would cost ~$0.10/query in LLM judge calls,
# making it prohibitively expensive at scale. Continuous eval also introduces latency.
# Weekly batch evaluation on a 50-question sample provides statistically meaningful
# signal for detecting quality drift while keeping costs under $5/week. For a document
# corpus that updates weekly (new RBI circulars, SEBI factsheets), weekly eval cadence
# matches the data change frequency — more frequent eval would mostly re-test stale data.

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from config.config import get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DriftReport(BaseModel):
    """The result of a single weekly drift-detection run.

    Attributes
    ----------
    timestamp:
        UTC timestamp when the report was generated.
    baseline_scores:
        Metric name -> score mapping representing the saved baseline.
    current_scores:
        Metric name -> score mapping from the latest evaluation run.
    drift_percentages:
        Metric name -> percentage change from baseline (positive = drop).
    alerts:
        Human-readable alert messages for metrics that breached their
        threshold.
    triggered_retrain:
        ``True`` when faithfulness dropped more than 10 %, causing an
        automatic retrain flag to be written.
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the drift report.",
    )
    baseline_scores: dict[str, float] = Field(
        description="Baseline metric scores loaded from Postgres.",
    )
    current_scores: dict[str, float] = Field(
        description="Current metric scores from the latest evaluation.",
    )
    drift_percentages: dict[str, float] = Field(
        description="Per-metric drift percentage (positive means degradation).",
    )
    alerts: list[str] = Field(
        default_factory=list,
        description="Human-readable alert strings for breached thresholds.",
    )
    triggered_retrain: bool = Field(
        default=False,
        description="Whether the drift was severe enough to trigger a retrain.",
    )


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_CREATE_BASELINE_TABLE = text("""
    CREATE TABLE IF NOT EXISTS eval_baseline (
        id              SERIAL PRIMARY KEY,
        metric_name     TEXT NOT NULL UNIQUE,
        score           DOUBLE PRECISION NOT NULL,
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
""")

_CREATE_DRIFT_REPORTS_TABLE = text("""
    CREATE TABLE IF NOT EXISTS drift_reports (
        id                  SERIAL PRIMARY KEY,
        report_timestamp    TIMESTAMPTZ NOT NULL,
        baseline_scores     JSONB NOT NULL,
        current_scores      JSONB NOT NULL,
        drift_percentages   JSONB NOT NULL,
        alerts              JSONB NOT NULL,
        triggered_retrain   BOOLEAN NOT NULL DEFAULT FALSE,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
""")

_CREATE_RETRAIN_TRIGGERS_TABLE = text("""
    CREATE TABLE IF NOT EXISTS retrain_triggers (
        id              SERIAL PRIMARY KEY,
        reason          TEXT NOT NULL,
        drift_report_id INTEGER,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        acknowledged    BOOLEAN NOT NULL DEFAULT FALSE
    )
""")

_UPSERT_BASELINE = text("""
    INSERT INTO eval_baseline (metric_name, score, updated_at)
    VALUES (:metric_name, :score, NOW())
    ON CONFLICT (metric_name)
    DO UPDATE SET score = EXCLUDED.score, updated_at = NOW()
""")

_SELECT_BASELINE = text("""
    SELECT metric_name, score FROM eval_baseline
""")

_INSERT_DRIFT_REPORT = text("""
    INSERT INTO drift_reports
        (report_timestamp, baseline_scores, current_scores,
         drift_percentages, alerts, triggered_retrain)
    VALUES
        (:ts, :baseline, :current, :drift, :alerts, :retrain)
    RETURNING id
""")

_INSERT_RETRAIN_TRIGGER = text("""
    INSERT INTO retrain_triggers (reason, drift_report_id)
    VALUES (:reason, :report_id)
""")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Detects quality drift by comparing weekly evaluation scores to a baseline.

    The detector relies on an *eval_suite* callable that, given a number of
    questions, returns a dict mapping metric names to their aggregate scores.
    This keeps the detector decoupled from any specific evaluation framework
    (e.g. RAGAS).

    Parameters
    ----------
    eval_suite:
        An async callable ``(num_questions: int) -> dict[str, float]`` that
        runs the evaluation pipeline on a random sample and returns metric
        scores.
    db_url:
        SQLAlchemy async connection URL for the Postgres database where
        baselines, drift reports, and retrain triggers are stored.  Defaults
        to the value of ``Settings.POSTGRES_URL``.
    """

    def __init__(
        self,
        eval_suite: Any,
        db_url: str | None = None,
    ) -> None:
        self._settings = get_settings()
        self._eval_suite = eval_suite
        self._db_url: str = db_url or self._settings.POSTGRES_URL
        self._engine = create_async_engine(
            self._db_url, pool_size=5, max_overflow=10,
        )
        logger.info("drift_detector.init", db_url=self._db_url)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def _ensure_tables(self) -> None:
        """Create required Postgres tables if they do not already exist."""

        async with self._engine.begin() as conn:
            await conn.execute(_CREATE_BASELINE_TABLE)
            await conn.execute(_CREATE_DRIFT_REPORTS_TABLE)
            await conn.execute(_CREATE_RETRAIN_TRIGGERS_TABLE)
        logger.debug("drift_detector.tables_ensured")

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    async def set_baseline(self, eval_report: dict[str, float]) -> None:
        """Persist an evaluation report as the new baseline.

        Each metric is upserted so that calling ``set_baseline`` twice
        with different sets of metrics will never lose older ones.

        Parameters
        ----------
        eval_report:
            A mapping of metric names (e.g. ``"faithfulness"``) to their
            baseline score values (0.0 -- 1.0).
        """

        await self._ensure_tables()

        async with AsyncSession(self._engine) as session:
            async with session.begin():
                for metric_name, score in eval_report.items():
                    await session.execute(
                        _UPSERT_BASELINE,
                        {"metric_name": metric_name, "score": float(score)},
                    )

        logger.info(
            "drift_detector.baseline_set",
            metrics=list(eval_report.keys()),
            scores=eval_report,
        )

    async def _load_baseline(self) -> dict[str, float]:
        """Load the current baseline scores from Postgres.

        Returns
        -------
        dict[str, float]
            Metric name -> score mapping.

        Raises
        ------
        RuntimeError
            If no baseline has been set yet.
        """

        async with AsyncSession(self._engine) as session:
            result = await session.execute(_SELECT_BASELINE)
            rows = result.mappings().all()

        if not rows:
            raise RuntimeError(
                "No baseline scores found in Postgres. "
                "Call set_baseline() before running a drift check."
            )

        baseline: dict[str, float] = {
            row["metric_name"]: float(row["score"]) for row in rows
        }
        logger.debug("drift_detector.baseline_loaded", baseline=baseline)
        return baseline

    # ------------------------------------------------------------------
    # Weekly check
    # ------------------------------------------------------------------

    async def run_weekly_check(self) -> DriftReport:
        """Execute the weekly drift-detection workflow.

        Steps
        -----
        1. Load baseline scores from Postgres.
        2. Run the evaluation suite on ``WEEKLY_EVAL_QUESTION_SAMPLE``
           questions.
        3. Compute per-metric drift percentage:
           ``drop_pct = (baseline - current) / baseline * 100``.
        4. For any metric where ``drop_pct`` exceeds the configured
           threshold, generate an alert string.
        5. If faithfulness specifically drops more than 10 %, flag a
           retrain.
        6. Persist the ``DriftReport`` to Postgres and, if any alerts
           were generated, fire a Slack notification.

        Returns
        -------
        DriftReport
            The complete drift report for the current run.
        """

        await self._ensure_tables()

        # Step 1: Load baseline
        baseline: dict[str, float] = await self._load_baseline()

        # Step 2: Run evaluation suite
        sample_size: int = self._settings.WEEKLY_EVAL_QUESTION_SAMPLE
        logger.info(
            "drift_detector.running_eval",
            sample_size=sample_size,
        )
        current_scores: dict[str, float] = await self._eval_suite(sample_size)

        # Step 3: Compute drift percentages
        threshold: float = self._settings.DRIFT_ALERT_THRESHOLD_PCT
        drift_percentages: dict[str, float] = {}
        alerts: list[str] = []
        triggered_retrain: bool = False

        for metric, baseline_score in baseline.items():
            current_score = current_scores.get(metric)
            if current_score is None:
                alerts.append(
                    f"Metric '{metric}' present in baseline but missing from "
                    f"current evaluation."
                )
                drift_percentages[metric] = 100.0
                continue

            if baseline_score == 0.0:
                # Avoid division by zero; treat any non-zero current as no drift
                drop_pct = 0.0 if current_score >= 0.0 else 100.0
            else:
                drop_pct = (baseline_score - current_score) / baseline_score * 100.0

            drift_percentages[metric] = round(drop_pct, 2)

            # Step 4: Threshold-based alerts
            if drop_pct > threshold:
                alert_msg = (
                    f"[DRIFT] {metric}: baseline={baseline_score:.4f} "
                    f"current={current_score:.4f} drop={drop_pct:.2f}% "
                    f"(threshold={threshold:.1f}%)"
                )
                alerts.append(alert_msg)
                logger.warning(
                    "drift_detector.threshold_breached",
                    metric=metric,
                    baseline=baseline_score,
                    current=current_score,
                    drop_pct=drop_pct,
                )

            # Step 5: Faithfulness retrain trigger
            if metric == "faithfulness" and drop_pct > 10.0:
                triggered_retrain = True
                logger.error(
                    "drift_detector.faithfulness_retrain",
                    drop_pct=drop_pct,
                )

        report = DriftReport(
            timestamp=datetime.now(timezone.utc),
            baseline_scores=baseline,
            current_scores=current_scores,
            drift_percentages=drift_percentages,
            alerts=alerts,
            triggered_retrain=triggered_retrain,
        )

        # Step 6: Persist report and conditionally alert
        report_id = await self._save_report(report)

        if triggered_retrain:
            await self._insert_retrain_trigger(
                reason=(
                    f"Faithfulness dropped {drift_percentages.get('faithfulness', 0):.2f}% "
                    f"(>10% threshold)"
                ),
                report_id=report_id,
            )

        if alerts:
            await self.send_slack_alert(report)

        logger.info(
            "drift_detector.weekly_check_complete",
            alert_count=len(alerts),
            triggered_retrain=triggered_retrain,
        )
        return report

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _save_report(self, report: DriftReport) -> int:
        """Write a ``DriftReport`` to the ``drift_reports`` table.

        Parameters
        ----------
        report:
            The drift report to persist.

        Returns
        -------
        int
            The auto-generated primary key of the inserted row.
        """

        async with AsyncSession(self._engine) as session:
            async with session.begin():
                result = await session.execute(
                    _INSERT_DRIFT_REPORT,
                    {
                        "ts": report.timestamp,
                        "baseline": json.dumps(report.baseline_scores),
                        "current": json.dumps(report.current_scores),
                        "drift": json.dumps(report.drift_percentages),
                        "alerts": json.dumps(report.alerts),
                        "retrain": report.triggered_retrain,
                    },
                )
                row = result.mappings().first()
                report_id: int = row["id"] if row else 0

        logger.info("drift_detector.report_saved", report_id=report_id)
        return report_id

    async def _insert_retrain_trigger(
        self,
        reason: str,
        report_id: int,
    ) -> None:
        """Write a retrain trigger row so downstream pipelines can pick it up.

        Parameters
        ----------
        reason:
            Human-readable explanation of why a retrain was triggered.
        report_id:
            Foreign key to the ``drift_reports`` row that caused the trigger.
        """

        async with AsyncSession(self._engine) as session:
            async with session.begin():
                await session.execute(
                    _INSERT_RETRAIN_TRIGGER,
                    {"reason": reason, "report_id": report_id},
                )

        logger.info(
            "drift_detector.retrain_trigger_inserted",
            reason=reason,
            report_id=report_id,
        )

    # ------------------------------------------------------------------
    # Slack alerting
    # ------------------------------------------------------------------

    async def send_slack_alert(self, report: DriftReport) -> None:
        """Post a drift-alert message to Slack via an incoming webhook.

        The message is formatted as a Slack Block Kit payload so it renders
        nicely in channels.  If ``SLACK_WEBHOOK_URL`` is not configured, the
        method logs a warning and returns without sending.

        Parameters
        ----------
        report:
            The ``DriftReport`` whose alerts should be sent.
        """

        webhook_url: str = self._settings.SLACK_WEBHOOK_URL
        if not webhook_url:
            logger.warning(
                "drift_detector.slack_alert_skipped",
                reason="SLACK_WEBHOOK_URL not configured",
            )
            return

        # Build a readable summary
        alert_lines = "\n".join(f"  * {a}" for a in report.alerts)
        retrain_note = (
            ":rotating_light: *Automatic retrain triggered* due to faithfulness drop."
            if report.triggered_retrain
            else ""
        )

        payload: dict[str, Any] = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "FinDocs Quality Drift Alert",
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Timestamp:* {report.timestamp.isoformat()}\n"
                            f"*Alerts ({len(report.alerts)}):*\n{alert_lines}"
                        ),
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"*{metric}:* baseline={bscore:.4f} "
                                f"current={report.current_scores.get(metric, 0.0):.4f} "
                                f"drift={report.drift_percentages.get(metric, 0.0):.2f}%"
                            ),
                        }
                        for metric, bscore in report.baseline_scores.items()
                    ],
                },
            ],
        }

        if retrain_note:
            payload["blocks"].append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": retrain_note},
                },
            )

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(webhook_url, json=payload)

        if response.status_code == 200:
            logger.info(
                "drift_detector.slack_alert_sent",
                alert_count=len(report.alerts),
            )
        else:
            logger.error(
                "drift_detector.slack_alert_failed",
                status_code=response.status_code,
                response_body=response.text[:500],
            )
