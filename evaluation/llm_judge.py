"""Custom LLM-as-judge for answer correctness scoring using Claude.

Prompts Anthropic Claude to evaluate how correct a RAG-generated answer is
relative to a ground-truth reference, returning a normalised 0-1 score
with chain-of-thought reasoning.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from findocs.config.config import Settings
    from findocs.evaluation.eval_suite import SingleEvalResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_JUDGE_SYSTEM_PROMPT: str = (
    "You are an expert evaluator for a financial document question-answering system. "
    "Your job is to judge how correct a generated answer is compared to the ground-truth "
    "reference answer.\n\n"
    "Scoring rubric (0.0 to 1.0):\n"
    "- 1.0: The answer is fully correct — it covers all key facts from the ground truth "
    "with no material errors.\n"
    "- 0.75: The answer is mostly correct — captures the main points but misses minor "
    "details or includes slight imprecisions.\n"
    "- 0.5: The answer is partially correct — some key facts are present but important "
    "information is missing or there are notable inaccuracies.\n"
    "- 0.25: The answer is mostly incorrect — only tangentially related to the ground "
    "truth or contains major errors.\n"
    "- 0.0: The answer is completely wrong, irrelevant, or empty.\n\n"
    "You MUST respond with valid JSON containing exactly two fields:\n"
    '  {"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}\n'
    "Do NOT include any other text outside the JSON object."
)

_JUDGE_USER_TEMPLATE: str = (
    "Question:\n{question}\n\n"
    "Ground-truth answer:\n{ground_truth}\n\n"
    "Generated answer:\n{answer}\n\n"
    "Evaluate the generated answer against the ground-truth. "
    "Respond ONLY with the JSON object."
)

_DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
_MAX_TOKENS: int = 512
_DEFAULT_RATE_LIMIT_DELAY: float = 0.5
_MAX_RETRIES: int = 3
_RETRY_BACKOFF_BASE: float = 2.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class JudgeResult(BaseModel):
    """Result from the LLM judge for a single answer evaluation."""

    score: float = Field(
        description="Correctness score between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Chain-of-thought explanation for the assigned score."
    )


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Uses Anthropic Claude as a judge to score answer correctness.

    Attributes:
        _client: Async Anthropic API client.
        _model: Model identifier to use for judging.
        _rate_limit_delay: Seconds to sleep between consecutive API calls
            when batch-judging to respect rate limits.
    """

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        *,
        model: str = _DEFAULT_MODEL,
        rate_limit_delay: float = _DEFAULT_RATE_LIMIT_DELAY,
    ) -> None:
        """Initialise the LLM judge.

        Args:
            anthropic_client: An ``AsyncAnthropic`` client instance.
            model: Anthropic model ID to use for judging.
            rate_limit_delay: Minimum delay in seconds between sequential
                API calls during batch judging.
        """
        self._client = anthropic_client
        self._model = model
        self._rate_limit_delay = rate_limit_delay

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMJudge:
        """Create an ``LLMJudge`` from application settings.

        Args:
            settings: FinDocs application settings containing the
                Anthropic API key.

        Returns:
            A configured ``LLMJudge`` instance.
        """
        client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        return cls(anthropic_client=client)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def judge_answer(
        self,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> JudgeResult:
        """Score a single answer against the ground truth using Claude.

        Sends a structured prompt to Claude asking it to evaluate the
        generated answer on a 0-1 correctness scale.  Retries with
        exponential back-off on transient failures.

        Args:
            question: The evaluation question.
            answer: The RAG-generated answer to evaluate.
            ground_truth: The expected reference answer.

        Returns:
            A ``JudgeResult`` with a score and reasoning.
        """
        log = logger.bind(question=question[:80])
        user_message = _JUDGE_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
        )

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    system=_JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )

                raw_text = response.content[0].text.strip()
                parsed = self._parse_judge_response(raw_text)
                log.debug(
                    "llm_judge.scored",
                    score=parsed.score,
                    attempt=attempt,
                )
                return parsed

            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
                log.warning(
                    "llm_judge.parse_error",
                    attempt=attempt,
                    error=str(exc),
                )
                last_error = exc
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF_BASE**attempt)

            except Exception as exc:
                log.warning(
                    "llm_judge.api_error",
                    attempt=attempt,
                    error=str(exc),
                )
                last_error = exc
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF_BASE**attempt)

        # All retries exhausted — return a zero score with error info
        log.error(
            "llm_judge.all_retries_failed",
            error=str(last_error),
        )
        return JudgeResult(
            score=0.0,
            reasoning=f"Judge failed after {_MAX_RETRIES} attempts: {last_error}",
        )

    async def batch_judge(
        self,
        results: list[SingleEvalResult],
    ) -> list[float]:
        """Judge multiple answers sequentially with rate limiting.

        Iterates over the provided results and calls ``judge_answer`` for
        each, inserting a configurable delay between calls to respect
        Anthropic API rate limits.

        Args:
            results: A list of ``SingleEvalResult`` objects containing the
                question, answer, and ground truth to evaluate.

        Returns:
            A list of correctness scores (0.0-1.0), one per input result,
            in the same order as the input.
        """
        log = logger.bind(batch_size=len(results))
        log.info("llm_judge.batch_judge.start")

        scores: list[float] = []
        for idx, result in enumerate(results):
            judge_result = await self.judge_answer(
                question=result.question,
                answer=result.answer,
                ground_truth=result.ground_truth,
            )
            scores.append(judge_result.score)

            log.debug(
                "llm_judge.batch_judge.progress",
                idx=idx + 1,
                total=len(results),
                score=judge_result.score,
            )

            # Rate-limit delay between calls (skip after the last one)
            if idx < len(results) - 1:
                await asyncio.sleep(self._rate_limit_delay)

        log.info(
            "llm_judge.batch_judge.done",
            mean_score=sum(scores) / len(scores) if scores else 0.0,
        )
        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_judge_response(raw_text: str) -> JudgeResult:
        """Parse the JSON response from Claude into a ``JudgeResult``.

        Handles cases where Claude wraps the JSON in markdown code fences.

        Args:
            raw_text: Raw text output from the Claude API.

        Returns:
            A validated ``JudgeResult``.

        Raises:
            json.JSONDecodeError: If the text cannot be parsed as JSON.
            ValueError: If the parsed score is outside [0, 1].
        """
        # Strip optional markdown code fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        score = float(data["score"])
        reasoning = str(data.get("reasoning", ""))

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Judge score {score} is outside the valid range [0.0, 1.0]"
            )

        return JudgeResult(score=score, reasoning=reasoning)
