"""Langfuse-backed prompt registry with caching and versioned rollback.

# ARCHITECTURE DECISION: Why Langfuse over git for prompt versioning
# Git-based prompt management requires a code deployment to change any prompt -- even
# a minor wording fix. Langfuse Prompt Management enables runtime hotswap: update a
# prompt in the Langfuse UI, and the next API request uses the new version with zero
# downtime. Crucially, each prompt version is linked to its eval scores, so you can
# compare "v3 had 0.82 faithfulness vs v4's 0.79" directly in the dashboard. This
# tight feedback loop between prompt changes and quality metrics is impossible with git.

Prompts are fetched from Langfuse with a short TTL cache (5 minutes) so that
hot-swapped prompts take effect almost immediately while avoiding excessive
API calls under high request concurrency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from langfuse import Langfuse

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Internal cache entry
# ---------------------------------------------------------------------------


class _CacheEntry:
    """In-memory TTL cache entry for a compiled prompt string."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: str, ttl: float) -> None:
        self.value: str = value
        self.expires_at: float = time.monotonic() + ttl

    def is_expired(self) -> bool:
        """Return ``True`` when the entry has exceeded its TTL."""
        return time.monotonic() >= self.expires_at


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """Manages versioned prompts via the Langfuse Prompt Management API.

    Provides a thin async layer over the Langfuse Python SDK that:

    * Fetches the latest (or a pinned) prompt version.
    * Substitutes template variables (``{question}``, ``{contexts}``, etc.).
    * Caches compiled prompts for ``_CACHE_TTL_SECONDS`` to reduce API load.

    Parameters
    ----------
    langfuse:
        An initialised ``Langfuse`` client instance.
    cache_ttl_s:
        Time-to-live for the in-memory prompt cache in seconds.
    """

    PROMPT_NAMES: dict[str, str] = {
        "rag_system": "findocs-rag-system",
        "rag_user": "findocs-rag-user",
        "query_classifier": "findocs-query-classifier",
    }
    """Mapping from logical prompt name to the Langfuse prompt name."""

    def __init__(
        self,
        langfuse: Langfuse,
        *,
        cache_ttl_s: float = _CACHE_TTL_SECONDS,
    ) -> None:
        self._langfuse: Langfuse = langfuse
        self._cache_ttl_s: float = cache_ttl_s
        self._cache: dict[str, _CacheEntry] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

        logger.info(
            "prompt_registry.init",
            prompt_names=list(self.PROMPT_NAMES.values()),
            cache_ttl_s=cache_ttl_s,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_prompt(
        self,
        name: str,
        variables: dict[str, str] | None = None,
    ) -> str:
        """Fetch the latest prompt version and compile it with variables.

        The compiled string is cached by ``(name, sorted-variables)`` for
        ``cache_ttl_s`` seconds.  On cache miss, the prompt is fetched from
        Langfuse in a thread-pool executor (the SDK is synchronous).

        Parameters
        ----------
        name:
            Logical prompt name -- must be a key in ``PROMPT_NAMES``.
        variables:
            Optional dict of template variables to substitute
            (e.g. ``{"question": "What is repo rate?"}``)

        Returns
        -------
        str
            The fully compiled prompt string.

        Raises
        ------
        KeyError
            If *name* is not found in ``PROMPT_NAMES``.
        langfuse.LangfuseError
            If the Langfuse API call fails.
        """

        langfuse_name = self._resolve_name(name)
        cache_key = self._build_cache_key(langfuse_name, variables)

        # Fast path: return from cache if still valid
        entry = self._cache.get(cache_key)
        if entry is not None and not entry.is_expired():
            logger.debug("prompt_registry.cache_hit", name=name)
            return entry.value

        # Slow path: fetch from Langfuse (synchronous SDK -> offload to thread)
        async with self._lock:
            # Double-check after acquiring the lock
            entry = self._cache.get(cache_key)
            if entry is not None and not entry.is_expired():
                return entry.value

            compiled = await self._fetch_and_compile(langfuse_name, variables)
            self._cache[cache_key] = _CacheEntry(compiled, self._cache_ttl_s)
            logger.info("prompt_registry.fetched", name=name, langfuse_name=langfuse_name)
            return compiled

    async def get_prompt_version(
        self,
        name: str,
        version: int,
        variables: dict[str, str] | None = None,
    ) -> str:
        """Fetch a specific prompt version for rollback or comparison.

        This method **always** calls Langfuse (no caching) because pinned-
        version lookups are typically low-frequency admin operations.

        Parameters
        ----------
        name:
            Logical prompt name (key in ``PROMPT_NAMES``).
        version:
            The integer version number to retrieve.
        variables:
            Optional template variables to substitute.

        Returns
        -------
        str
            The compiled prompt string for the requested version.
        """

        langfuse_name = self._resolve_name(name)

        loop = asyncio.get_running_loop()
        prompt_obj = await loop.run_in_executor(
            None,
            lambda: self._langfuse.get_prompt(langfuse_name, version=version),
        )

        template: str = prompt_obj.prompt  # type: ignore[union-attr]
        compiled = self._substitute(template, variables)

        logger.info(
            "prompt_registry.version_fetched",
            name=name,
            version=version,
        )
        return compiled

    def get_all_prompt_versions(self, name: str) -> list[dict[str, Any]]:
        """Return metadata for every version of a prompt.

        Calls the Langfuse SDK synchronously (intended for admin/CLI use).

        Parameters
        ----------
        name:
            Logical prompt name (key in ``PROMPT_NAMES``).

        Returns
        -------
        list[dict[str, Any]]
            A list of dicts, each containing ``version``, ``prompt``
            (the raw template), ``created_at``, and ``labels``.
        """

        langfuse_name = self._resolve_name(name)

        # The Langfuse SDK exposes prompt listing via get_prompt with
        # fetch_all or via the underlying API.  We iterate known versions
        # by fetching the latest and walking backwards until a 404.
        versions: list[dict[str, Any]] = []
        version_num = 1

        while True:
            try:
                prompt_obj = self._langfuse.get_prompt(langfuse_name, version=version_num)
            except Exception:
                # Reached beyond the highest version
                break

            versions.append(
                {
                    "version": version_num,
                    "prompt": prompt_obj.prompt,
                    "created_at": getattr(prompt_obj, "created_at", None),
                    "labels": getattr(prompt_obj, "labels", []),
                }
            )
            version_num += 1

        logger.info(
            "prompt_registry.listed_versions",
            name=name,
            count=len(versions),
        )
        return versions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_name(self, name: str) -> str:
        """Map a logical name to the Langfuse prompt name.

        Parameters
        ----------
        name:
            Key from ``PROMPT_NAMES`` or a raw Langfuse name.

        Returns
        -------
        str
            The Langfuse-side prompt name.

        Raises
        ------
        KeyError
            If *name* is not in ``PROMPT_NAMES`` and does not match any
            registered Langfuse name.
        """

        if name in self.PROMPT_NAMES:
            return self.PROMPT_NAMES[name]

        # Allow passing the Langfuse name directly
        if name in self.PROMPT_NAMES.values():
            return name

        raise KeyError(
            f"Unknown prompt name '{name}'. "
            f"Registered names: {list(self.PROMPT_NAMES.keys())}"
        )

    async def _fetch_and_compile(
        self,
        langfuse_name: str,
        variables: dict[str, str] | None,
    ) -> str:
        """Fetch the latest prompt from Langfuse and compile it.

        Parameters
        ----------
        langfuse_name:
            The prompt name as registered in Langfuse.
        variables:
            Optional template variables.

        Returns
        -------
        str
            The compiled prompt string.
        """

        loop = asyncio.get_running_loop()
        prompt_obj = await loop.run_in_executor(
            None,
            lambda: self._langfuse.get_prompt(langfuse_name),
        )

        template: str = prompt_obj.prompt  # type: ignore[union-attr]
        return self._substitute(template, variables)

    @staticmethod
    def _substitute(template: str, variables: dict[str, str] | None) -> str:
        """Replace ``{variable}`` placeholders in a template string.

        Parameters
        ----------
        template:
            Raw template text with ``{placeholder}`` markers.
        variables:
            Mapping of placeholder name to replacement value.

        Returns
        -------
        str
            The template with all known placeholders substituted.
        """

        if not variables:
            return template

        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    @staticmethod
    def _build_cache_key(
        langfuse_name: str,
        variables: dict[str, str] | None,
    ) -> str:
        """Build a deterministic cache key from prompt name and variables.

        Parameters
        ----------
        langfuse_name:
            Langfuse prompt name.
        variables:
            Template variables (order-insensitive).

        Returns
        -------
        str
            A string key suitable for dict-based caching.
        """

        if not variables:
            return langfuse_name

        sorted_vars = "|".join(f"{k}={v}" for k, v in sorted(variables.items()))
        return f"{langfuse_name}::{sorted_vars}"
