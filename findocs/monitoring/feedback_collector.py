"""User feedback collection and storage.

Collects thumbs-up/down feedback on RAG responses and persists to Postgres.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from findocs.config.config import get_settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger()
settings = get_settings()


class FeedbackRecord(BaseModel):
    """A single user feedback entry."""

    trace_id: str
    session_id: str | None = None
    question: str
    answer: str
    score: int = Field(ge=-1, le=1)  # -1 = thumbs down, 0 = neutral, 1 = thumbs up
    comment: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackSummary(BaseModel):
    """Aggregated feedback statistics."""

    total_feedback: int
    positive: int
    negative: int
    neutral: int
    positive_rate: float
    avg_score: float
    period_start: datetime | None = None
    period_end: datetime | None = None


class FeedbackCollector:
    """Collects, stores, and queries user feedback on RAG responses."""

    def __init__(self, postgres_url: str | None = None) -> None:
        """Initialise with Postgres connection.

        Args:
            postgres_url: Async Postgres connection URL.
        """
        self.engine: AsyncEngine = create_async_engine(
            postgres_url or settings.POSTGRES_URL,
            pool_size=5,
            max_overflow=10,
        )

    async def store_feedback(self, feedback: FeedbackRecord) -> str:
        """Store a feedback record in Postgres.

        Args:
            feedback: The feedback record to store.

        Returns:
            The ID of the stored feedback record.
        """
        query = text(
            """
            INSERT INTO user_feedback (trace_id, session_id, question, answer, score, comment, created_at)
            VALUES (:trace_id, :session_id, :question, :answer, :score, :comment, :created_at)
            RETURNING id::text
            """
        )

        async with self.engine.begin() as conn:
            result = await conn.execute(
                query,
                {
                    "trace_id": feedback.trace_id,
                    "session_id": feedback.session_id,
                    "question": feedback.question,
                    "answer": feedback.answer,
                    "score": feedback.score,
                    "comment": feedback.comment,
                    "created_at": feedback.created_at,
                },
            )
            row = result.fetchone()
            feedback_id: str = row[0] if row else ""

        logger.info(
            "feedback_stored",
            feedback_id=feedback_id,
            trace_id=feedback.trace_id,
            score=feedback.score,
        )
        return feedback_id

    async def get_feedback_by_trace(self, trace_id: str) -> list[FeedbackRecord]:
        """Retrieve all feedback for a specific trace.

        Args:
            trace_id: The Langfuse trace ID.

        Returns:
            List of feedback records for the trace.
        """
        query = text(
            """
            SELECT trace_id, session_id, question, answer, score, comment, created_at
            FROM user_feedback
            WHERE trace_id = :trace_id
            ORDER BY created_at DESC
            """
        )

        async with self.engine.connect() as conn:
            result = await conn.execute(query, {"trace_id": trace_id})
            rows = result.fetchall()

        return [
            FeedbackRecord(
                trace_id=row[0],
                session_id=row[1],
                question=row[2],
                answer=row[3],
                score=row[4],
                comment=row[5],
                created_at=row[6],
            )
            for row in rows
        ]

    async def get_summary(
        self,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
    ) -> FeedbackSummary:
        """Get aggregated feedback statistics for a time period.

        Args:
            period_start: Start of the period (inclusive). Defaults to all time.
            period_end: End of the period (inclusive). Defaults to now.

        Returns:
            Aggregated feedback summary.
        """
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if period_start:
            conditions.append("created_at >= :period_start")
            params["period_start"] = period_start
        if period_end:
            conditions.append("created_at <= :period_end")
            params["period_end"] = period_end

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = text(
            f"""
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END), 0) as positive,
                COALESCE(SUM(CASE WHEN score = -1 THEN 1 ELSE 0 END), 0) as negative,
                COALESCE(SUM(CASE WHEN score = 0 THEN 1 ELSE 0 END), 0) as neutral,
                COALESCE(AVG(score), 0) as avg_score
            FROM user_feedback
            {where_clause}
            """
        )

        async with self.engine.connect() as conn:
            result = await conn.execute(query, params)
            row = result.fetchone()

        if not row or row[0] == 0:
            return FeedbackSummary(
                total_feedback=0,
                positive=0,
                negative=0,
                neutral=0,
                positive_rate=0.0,
                avg_score=0.0,
                period_start=period_start,
                period_end=period_end,
            )

        total = int(row[0])
        positive = int(row[1])

        return FeedbackSummary(
            total_feedback=total,
            positive=positive,
            negative=int(row[2]),
            neutral=int(row[3]),
            positive_rate=positive / total if total > 0 else 0.0,
            avg_score=float(row[4]),
            period_start=period_start,
            period_end=period_end,
        )

    async def get_negative_feedback(self, limit: int = 50) -> list[FeedbackRecord]:
        """Retrieve recent negative feedback for review.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of negative feedback records, most recent first.
        """
        query = text(
            """
            SELECT trace_id, session_id, question, answer, score, comment, created_at
            FROM user_feedback
            WHERE score = -1
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )

        async with self.engine.connect() as conn:
            result = await conn.execute(query, {"limit": limit})
            rows = result.fetchall()

        return [
            FeedbackRecord(
                trace_id=row[0],
                session_id=row[1],
                question=row[2],
                answer=row[3],
                score=row[4],
                comment=row[5],
                created_at=row[6],
            )
            for row in rows
        ]
