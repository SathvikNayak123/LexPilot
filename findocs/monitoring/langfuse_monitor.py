"""Production query tracing and dashboard statistics via Langfuse.

Wraps the Langfuse SDK to provide structured trace creation, retrieval/generation
span logging, user-feedback capture, and daily aggregate statistics for the
FinDocs RAG pipeline.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any

import structlog
from langfuse import Langfuse

from config.config import get_settings

logger = structlog.get_logger(__name__)


class LangfuseMonitor:
    """High-level tracing facade over the Langfuse Python SDK.

    Each user query creates a *trace* that groups a retrieval span and a
    generation span.  User feedback is recorded as a Langfuse *score* attached
    to the trace.  ``get_daily_stats`` returns aggregate numbers suitable for
    a lightweight operational dashboard.

    Parameters
    ----------
    langfuse:
        An already-initialised ``Langfuse`` client instance.  Typically
        constructed from ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``,
        and ``LANGFUSE_HOST`` in the application settings.
    """

    def __init__(self, langfuse: Langfuse) -> None:
        self._langfuse: Langfuse = langfuse
        self._settings = get_settings()
        logger.info(
            "langfuse_monitor.init",
            host=self._settings.LANGFUSE_HOST,
        )

    # ------------------------------------------------------------------
    # Trace creation
    # ------------------------------------------------------------------

    def create_trace(
        self,
        question: str,
        session_id: str | None = None,
    ) -> Any:
        """Create a new Langfuse trace for an incoming user question.

        A trace is the top-level unit that groups all spans (retrieval,
        generation) and scores for a single query.

        Parameters
        ----------
        question:
            The raw user question that initiated the query.
        session_id:
            An optional session identifier used to group multiple
            traces belonging to the same user conversation.

        Returns
        -------
        object
            A Langfuse ``StatefulTraceClient`` that can be used with
            ``log_retrieval`` and ``log_generation`` to add child spans.
        """

        trace = self._langfuse.trace(
            name="findocs-query",
            input={"question": question},
            session_id=session_id,
            metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info(
            "langfuse_monitor.trace_created",
            trace_id=trace.id,
            session_id=session_id,
            question_length=len(question),
        )
        return trace

    # ------------------------------------------------------------------
    # Retrieval span
    # ------------------------------------------------------------------

    def log_retrieval(
        self,
        trace: Any,
        contexts: list[str],
        latency_ms: float,
        scores: dict[str, float],
    ) -> None:
        """Log the retrieval step as a span on an existing trace.

        Parameters
        ----------
        trace:
            The parent ``StatefulTraceClient`` returned by ``create_trace``.
        contexts:
            The list of text chunks retrieved from the vector store.
        latency_ms:
            Wall-clock time in milliseconds for the retrieval step.
        scores:
            Named retrieval-quality scores (e.g. ``{"mrr": 0.85}``).
        """

        span = trace.span(
            name="retrieval",
            input={"trace_id": trace.id},
            output={"contexts": contexts, "num_contexts": len(contexts)},
            metadata={
                "latency_ms": latency_ms,
                "scores": scores,
            },
        )

        # Attach each retrieval score to the span for Langfuse dashboards
        for score_name, score_value in scores.items():
            span.score(
                name=f"retrieval_{score_name}",
                value=score_value,
            )

        logger.info(
            "langfuse_monitor.retrieval_logged",
            trace_id=trace.id,
            num_contexts=len(contexts),
            latency_ms=latency_ms,
            scores=scores,
        )

    # ------------------------------------------------------------------
    # Generation span
    # ------------------------------------------------------------------

    def log_generation(
        self,
        trace: Any,
        prompt: str,
        response: str,
        model: str,
        tokens: dict[str, int],
        latency_ms: float,
        cost: float,
    ) -> None:
        """Log the LLM generation step on an existing trace.

        Parameters
        ----------
        trace:
            The parent ``StatefulTraceClient`` returned by ``create_trace``.
        prompt:
            The full prompt sent to the LLM (system + user message).
        response:
            The raw text response from the LLM.
        model:
            Model identifier (e.g. ``"gpt-4o-mini"`` or ``"phi3-findocs"``).
        tokens:
            Token usage dict with keys ``prompt_tokens`` and
            ``completion_tokens``.
        latency_ms:
            Wall-clock time in milliseconds for the generation call.
        cost:
            Estimated cost in USD for this generation call.
        """

        trace.generation(
            name="generation",
            model=model,
            input={"prompt": prompt},
            output={"response": response},
            usage={
                "input": tokens.get("prompt_tokens", 0),
                "output": tokens.get("completion_tokens", 0),
                "total": (
                    tokens.get("prompt_tokens", 0)
                    + tokens.get("completion_tokens", 0)
                ),
            },
            metadata={
                "latency_ms": latency_ms,
                "cost_usd": cost,
            },
        )

        logger.info(
            "langfuse_monitor.generation_logged",
            trace_id=trace.id,
            model=model,
            prompt_tokens=tokens.get("prompt_tokens", 0),
            completion_tokens=tokens.get("completion_tokens", 0),
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    # ------------------------------------------------------------------
    # User feedback
    # ------------------------------------------------------------------

    def log_feedback(
        self,
        trace_id: str,
        score: float,
        comment: str | None = None,
    ) -> None:
        """Log user feedback as a Langfuse score on a trace.

        Parameters
        ----------
        trace_id:
            The identifier of the trace to attach feedback to.
        score:
            Numeric feedback value (e.g. ``1`` for thumbs-up, ``-1`` for
            thumbs-down, or a continuous rating).
        comment:
            Optional free-text comment from the user.
        """

        self._langfuse.score(
            trace_id=trace_id,
            name="user_feedback",
            value=score,
            comment=comment,
        )

        logger.info(
            "langfuse_monitor.feedback_logged",
            trace_id=trace_id,
            score=score,
            has_comment=comment is not None,
        )

    # ------------------------------------------------------------------
    # Daily statistics
    # ------------------------------------------------------------------

    async def get_daily_stats(self, target_date: date) -> dict[str, Any]:
        """Query Langfuse for aggregate statistics on a given day.

        Fetches all traces for ``target_date``, then computes summary
        metrics suitable for a lightweight operational dashboard.

        Parameters
        ----------
        target_date:
            The calendar date to aggregate statistics for.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:
            - ``date`` -- the target date as an ISO string.
            - ``trace_count`` -- total number of traces.
            - ``avg_latency_ms`` -- mean end-to-end latency.
            - ``avg_generation_latency_ms`` -- mean generation latency.
            - ``avg_retrieval_latency_ms`` -- mean retrieval latency.
            - ``avg_user_feedback`` -- mean user feedback score.
            - ``total_cost_usd`` -- summed generation cost.
            - ``model_usage`` -- dict mapping model names to usage counts.
        """

        start_dt = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            tzinfo=timezone.utc,
        )
        end_dt = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            23, 59, 59,
            tzinfo=timezone.utc,
        )

        # Langfuse SDK fetch_traces is synchronous; we wrap it so callers can
        # treat this method as async (consistent with the rest of the codebase).
        traces_response = self._langfuse.fetch_traces(
            from_timestamp=start_dt,
            to_timestamp=end_dt,
            limit=10_000,
        )
        traces = traces_response.data

        if not traces:
            logger.info(
                "langfuse_monitor.daily_stats.no_traces",
                date=target_date.isoformat(),
            )
            return {
                "date": target_date.isoformat(),
                "trace_count": 0,
                "avg_latency_ms": 0.0,
                "avg_generation_latency_ms": 0.0,
                "avg_retrieval_latency_ms": 0.0,
                "avg_user_feedback": 0.0,
                "total_cost_usd": 0.0,
                "model_usage": {},
            }

        # Accumulate metrics across all traces
        latencies: list[float] = []
        generation_latencies: list[float] = []
        retrieval_latencies: list[float] = []
        feedback_scores: list[float] = []
        total_cost: float = 0.0
        model_usage: dict[str, int] = {}

        for trace in traces:
            # Observations (spans / generations) attached to this trace
            observations = self._langfuse.fetch_observations(
                trace_id=trace.id,
                limit=100,
            ).data

            trace_start: float | None = None
            trace_end: float | None = None

            for obs in observations:
                obs_meta = obs.metadata or {}

                # Latency from metadata
                obs_latency = obs_meta.get("latency_ms")

                if obs.name == "retrieval" and obs_latency is not None:
                    retrieval_latencies.append(float(obs_latency))

                if obs.name == "generation":
                    if obs_latency is not None:
                        generation_latencies.append(float(obs_latency))
                    cost_val = obs_meta.get("cost_usd", 0.0)
                    total_cost += float(cost_val)

                    # Track model usage
                    model_name = getattr(obs, "model", None) or "unknown"
                    model_usage[model_name] = model_usage.get(model_name, 0) + 1

                # Determine overall trace latency from observation timestamps
                if obs.start_time is not None:
                    start_ts = obs.start_time.timestamp()
                    if trace_start is None or start_ts < trace_start:
                        trace_start = start_ts
                if obs.end_time is not None:
                    end_ts = obs.end_time.timestamp()
                    if trace_end is None or end_ts > trace_end:
                        trace_end = end_ts

            if trace_start is not None and trace_end is not None:
                latencies.append((trace_end - trace_start) * 1000.0)

            # Scores attached to the trace
            scores_response = self._langfuse.fetch_scores(
                trace_id=trace.id,
                limit=100,
            )
            for score_obj in scores_response.data:
                if score_obj.name == "user_feedback" and score_obj.value is not None:
                    feedback_scores.append(float(score_obj.value))

        def _safe_mean(values: list[float]) -> float:
            """Return arithmetic mean or 0.0 for empty lists."""
            return sum(values) / len(values) if values else 0.0

        stats: dict[str, Any] = {
            "date": target_date.isoformat(),
            "trace_count": len(traces),
            "avg_latency_ms": round(_safe_mean(latencies), 2),
            "avg_generation_latency_ms": round(_safe_mean(generation_latencies), 2),
            "avg_retrieval_latency_ms": round(_safe_mean(retrieval_latencies), 2),
            "avg_user_feedback": round(_safe_mean(feedback_scores), 4),
            "total_cost_usd": round(total_cost, 4),
            "model_usage": model_usage,
        }

        logger.info(
            "langfuse_monitor.daily_stats",
            date=target_date.isoformat(),
            trace_count=stats["trace_count"],
            avg_latency_ms=stats["avg_latency_ms"],
            avg_user_feedback=stats["avg_user_feedback"],
        )
        return stats
