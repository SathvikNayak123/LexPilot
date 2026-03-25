"""Fine-tuned model to GPT-4o fallback chain.

Implements a two-tier generation strategy: the primary (self-hosted) model is
tried first, and on timeout or error the request is transparently retried
against an OpenAI-hosted fallback model.  Every call returns both the
generation result and a tag identifying which model actually served the
request, so the caller can log model provenance in traces.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from openai import AsyncOpenAI

from findocs.serving.model_server import GenerationResult, ModelServer

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class FallbackChain:
    """Two-tier generation chain: self-hosted primary -> OpenAI fallback.

    Parameters
    ----------
    primary:
        A ``ModelServer`` instance pointing at the fine-tuned model
        (e.g. Phi-3 served via vLLM or Ollama).
    fallback_client:
        An ``AsyncOpenAI`` client configured with an API key.
    fallback_model:
        The OpenAI model name to use when the primary is unavailable
        (e.g. ``"gpt-4o"`` or ``"gpt-4o-mini"``).
    """

    def __init__(
        self,
        primary: ModelServer,
        fallback_client: AsyncOpenAI,
        fallback_model: str = "gpt-4o",
    ) -> None:
        self._primary: ModelServer = primary
        self._fallback_client: AsyncOpenAI = fallback_client
        self._fallback_model: str = fallback_model

        logger.info(
            "fallback_chain.init",
            primary_model=primary._model_name,
            fallback_model=fallback_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
    ) -> tuple[GenerationResult, str]:
        """Generate text, falling back to OpenAI on primary failure.

        Parameters
        ----------
        messages:
            OpenAI-style chat messages (``role`` / ``content`` dicts).
        max_tokens:
            Maximum tokens to generate.

        Returns
        -------
        tuple[GenerationResult, str]
            A 2-tuple of ``(result, model_used)`` where ``model_used`` is
            either ``"primary"`` or ``"fallback"``.
        """

        # --- Attempt 1: primary (fine-tuned, self-hosted) -----------------
        try:
            result = await self._primary.generate(
                messages,
                max_tokens=max_tokens,
                temperature=0.1,
            )
            logger.info(
                "fallback_chain.primary_success",
                model=result.model,
                latency_ms=result.latency_ms,
            )
            return result, "primary"

        except Exception as primary_exc:
            logger.warning(
                "fallback_chain.primary_failed",
                error=str(primary_exc),
                error_type=type(primary_exc).__name__,
            )

        # --- Attempt 2: fallback (OpenAI) ---------------------------------
        try:
            result = await self._call_openai_fallback(messages, max_tokens=max_tokens)
            logger.info(
                "fallback_chain.fallback_success",
                model=result.model,
                latency_ms=result.latency_ms,
            )
            return result, "fallback"

        except Exception as fallback_exc:
            logger.error(
                "fallback_chain.fallback_failed",
                error=str(fallback_exc),
                error_type=type(fallback_exc).__name__,
            )
            raise fallback_exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_openai_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """Call the OpenAI chat-completions API as a fallback.

        Parameters
        ----------
        messages:
            Chat messages in OpenAI format.
        max_tokens:
            Maximum tokens to generate.

        Returns
        -------
        GenerationResult
            Parsed result with content, model tag, token count, and latency.

        Raises
        ------
        openai.APIError
            If the OpenAI API returns an error.
        """

        start = time.perf_counter()

        response: Any = await self._fallback_client.chat.completions.create(
            model=self._fallback_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0.1,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        choice = response.choices[0]
        content: str = choice.message.content or ""
        usage = response.usage
        tokens_used: int = usage.total_tokens if usage else 0

        return GenerationResult(
            content=content,
            model=response.model or self._fallback_model,
            tokens_used=tokens_used,
            latency_ms=round(elapsed_ms, 2),
        )
