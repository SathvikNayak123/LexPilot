"""POST /feedback – user feedback endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from findocs.api.schemas import FeedbackRequest, FeedbackResponse
from findocs.monitoring.feedback_collector import FeedbackCollector, FeedbackRecord

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

router = APIRouter()

_collector: FeedbackCollector | None = None


def _get_collector() -> FeedbackCollector:
    """Build or return the cached FeedbackCollector singleton."""
    global _collector  # noqa: PLW0603
    if _collector is None:
        _collector = FeedbackCollector()
    return _collector


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Store user feedback for a RAG response.

    Args:
        request: Feedback payload with trace_id, score, and optional comment.

    Returns:
        Confirmation with the stored feedback ID.
    """
    collector = _get_collector()

    record = FeedbackRecord(
        trace_id=request.trace_id,
        session_id=request.session_id,
        question=request.question,
        answer=request.answer,
        score=request.score,
        comment=request.comment,
    )

    try:
        feedback_id = await collector.store_feedback(record)
    except Exception as exc:
        logger.error("feedback_storage_error", error=str(exc), trace_id=request.trace_id)
        raise HTTPException(status_code=500, detail="Failed to store feedback") from exc

    return FeedbackResponse(feedback_id=feedback_id, status="stored")
