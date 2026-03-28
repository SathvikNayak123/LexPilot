from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

from config.config import settings

router = APIRouter(tags=["feedback"])

engine = create_async_engine(settings.postgres_url)


class FeedbackRequest(BaseModel):
    session_id: str
    trace_id: Optional[str] = None
    rating: int  # -1 or 1
    comment: Optional[str] = None


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Store user feedback for a research session."""
    async with AsyncSession(engine) as session:
        await session.execute(
            text("""
                INSERT INTO feedback (session_id, trace_id, rating, comment)
                VALUES (:sid, :tid, :rating, :comment)
            """),
            {
                "sid": req.session_id,
                "tid": req.trace_id,
                "rating": req.rating,
                "comment": req.comment,
            },
        )
        await session.commit()
    return {"status": "ok"}
