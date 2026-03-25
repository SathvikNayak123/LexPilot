"""vLLM / Ollama model server client with OpenAI-compatible API.

Provides an async client for self-hosted language models exposed through an
OpenAI-compatible HTTP endpoint (e.g. vLLM ``--api-server``, Ollama, or
LocalAI).  The client handles timeout, retries, and structured result
extraction so that callers never deal with raw HTTP responses.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class GenerationResult(BaseModel):
    """Structured result returned by the model server after generation."""

    content: str = Field(description="Generated text content.")
    model: str = Field(description="Name of the model that produced the output.")
    tokens_used: int = Field(default=0, description="Total tokens consumed (prompt + completion).")
    latency_ms: float = Field(default=0.0, description="Wall-clock latency of the generation call in milliseconds.")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ModelServer:
    """Async client for a self-hosted model behind an OpenAI-compatible API.

    Parameters
    ----------
    base_url:
        Root URL of the model server (e.g. ``http://localhost:8000/v1``).
    model_name:
        Model identifier to pass in the ``model`` field of chat-completion
        requests.
    timeout_s:
        Per-request timeout in seconds.
    max_retries:
        Number of retry attempts on transient failures before raising.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._base_url: str = base_url.rstrip("/")
        self._model_name: str = model_name
        self._timeout_s: float = timeout_s
        self._max_retries: int = max_retries
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s),
        )

        logger.info(
            "model_server.init",
            base_url=self._base_url,
            model=self._model_name,
            timeout_s=timeout_s,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> GenerationResult:
        """Send a chat-completion request to the model server.

        Parameters
        ----------
        messages:
            OpenAI-style message list (each dict has ``role`` and ``content``).
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature.

        Returns
        -------
        GenerationResult
            Parsed generation output including content, model name, token
            usage, and latency.

        Raises
        ------
        httpx.HTTPStatusError
            If the server returns a non-2xx status after all retries.
        httpx.TimeoutException
            If the request exceeds the configured timeout.
        """

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_exc: BaseException | None = None

        for attempt in range(1, self._max_retries + 1):
            start = time.perf_counter()
            try:
                response = await self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                data: dict[str, Any] = response.json()
                choice = data["choices"][0]
                content: str = choice["message"]["content"]

                usage: dict[str, int] = data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)

                result = GenerationResult(
                    content=content,
                    model=data.get("model", self._model_name),
                    tokens_used=tokens_used,
                    latency_ms=round(elapsed_ms, 2),
                )

                logger.info(
                    "model_server.generate.success",
                    model=result.model,
                    tokens_used=result.tokens_used,
                    latency_ms=result.latency_ms,
                    attempt=attempt,
                )
                return result

            except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning(
                    "model_server.generate.retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt >= self._max_retries:
                    break

        logger.error(
            "model_server.generate.failed",
            model=self._model_name,
            error=str(last_exc),
        )
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Probe the model server's health endpoint.

        Tries ``GET /health`` first (vLLM convention), then falls back to
        ``GET /v1/models`` (OpenAI-compatible convention).

        Returns
        -------
        bool
            ``True`` if the server responds with a 2xx status, ``False``
            otherwise.
        """

        for path in ("/health", "/v1/models"):
            try:
                response = await self._client.get(path)
                if response.is_success:
                    logger.info("model_server.health_check.ok", path=path)
                    return True
            except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException) as exc:
                logger.debug("model_server.health_check.probe_failed", path=path, error=str(exc))

        logger.warning("model_server.health_check.unhealthy", base_url=self._base_url)
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
        logger.info("model_server.closed")
