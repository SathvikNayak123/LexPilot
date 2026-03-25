"""Chart/figure description extraction using OpenAI GPT-4o Vision.

Renders chart regions from PDF pages as PNG images, sends them to the
GPT-4o Vision API, and returns structured ``ChartBlock`` descriptions.

Includes per-PDF rate limiting (max 10 charts) and a cumulative cost
guard ($0.50 per PDF) to control spend.
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING

import fitz
import structlog
from openai import AsyncOpenAI

from findocs.ingestion.models import ChartBlock

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Cost estimation helpers
# ---------------------------------------------------------------------------
# Approximate pricing for GPT-4o vision requests (as of 2025).
# Input: ~$2.50 / 1M tokens; images are counted as ~765 tokens for a
# 512x512 low-detail tile.  We use a conservative flat estimate per call.
_ESTIMATED_COST_PER_CALL_USD: float = 0.01

# ---------------------------------------------------------------------------
# Vision prompt
# ---------------------------------------------------------------------------
_VISION_SYSTEM_PROMPT: str = (
    "You are a financial-document analyst.  Describe the chart or figure in "
    "the provided image.  Return a JSON object (no markdown fences) with keys: "
    '"chart_type" (e.g. bar, line, pie, scatter, waterfall, area, table-figure), '
    '"title" (chart title if visible, else null), '
    '"axis_labels" (object with "x" and "y" keys, null if not applicable), '
    '"key_data_points" (list of the most important numeric values or labels), '
    '"summary" (one-paragraph natural-language description of what the chart shows).'
)


class ChartExtractor:
    """Extract chart/figure descriptions from PDF pages via GPT-4o Vision.

    Attributes:
        max_charts_per_pdf: Maximum number of chart extractions allowed per
            PDF to limit API calls.
        cost_limit_usd: Cumulative cost ceiling (USD) per PDF; extraction
            stops once this is exceeded.
    """

    max_charts_per_pdf: int = 10
    cost_limit_usd: float = 0.50

    def __init__(self, openai_api_key: str | None = None) -> None:
        """Initialise the extractor.

        Args:
            openai_api_key: Optional OpenAI API key.  When ``None`` the
                ``AsyncOpenAI`` client falls back to the ``OPENAI_API_KEY``
                environment variable.
        """
        self._client = AsyncOpenAI(api_key=openai_api_key)
        self._pdf_call_count: int = 0
        self._pdf_total_cost: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Reset per-PDF rate-limit and cost counters.

        Call this before starting extraction on a new PDF.
        """
        self._pdf_call_count = 0
        self._pdf_total_cost = 0.0

    async def extract_chart_description(
        self,
        page: fitz.Page,
        image_rect: fitz.Rect,
    ) -> ChartBlock | None:
        """Render a page region to PNG and describe it via GPT-4o Vision.

        Checks the per-PDF rate limit and cost guard before making the
        API call.  Returns ``None`` if either limit has been reached.

        Args:
            page: A PyMuPDF ``fitz.Page`` object.
            image_rect: Bounding rectangle of the chart/figure on the page.

        Returns:
            A ``ChartBlock`` with the LLM-generated description, or ``None``
            if limits are exceeded or the API call fails.
        """
        page_num = page.number + 1  # 1-indexed

        # Rate-limit guard
        if self._pdf_call_count >= self.max_charts_per_pdf:
            logger.warning(
                "chart_extractor.rate_limit_reached",
                page_num=page_num,
                max_charts=self.max_charts_per_pdf,
            )
            return None

        # Cost guard
        if self._pdf_total_cost >= self.cost_limit_usd:
            logger.warning(
                "chart_extractor.cost_limit_reached",
                page_num=page_num,
                total_cost=self._pdf_total_cost,
                limit=self.cost_limit_usd,
            )
            return None

        # ----- Render region to PNG ----- #
        try:
            clip = fitz.Rect(image_rect)
            # Use a 2x zoom for better quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            png_bytes: bytes = pix.tobytes("png")
        except Exception:
            logger.exception(
                "chart_extractor.render_error",
                page_num=page_num,
                rect=tuple(image_rect),
            )
            return None

        b64_image = base64.b64encode(png_bytes).decode("ascii")

        # ----- Call GPT-4o Vision ----- #
        try:
            response = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "low",
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe this chart from a financial document. "
                                    "Include chart type, key numbers, axis labels, "
                                    "and a brief summary."
                                ),
                            },
                        ],
                    },
                ],
                max_tokens=512,
                temperature=0.2,
            )

            description = response.choices[0].message.content or ""

            # Estimate cost from token usage if available
            usage = response.usage
            if usage is not None:
                prompt_cost = (usage.prompt_tokens / 1_000_000) * 2.50
                completion_cost = (usage.completion_tokens / 1_000_000) * 10.00
                call_cost = prompt_cost + completion_cost
            else:
                call_cost = _ESTIMATED_COST_PER_CALL_USD

        except Exception:
            logger.exception(
                "chart_extractor.api_error",
                page_num=page_num,
            )
            return None

        # Update counters
        self._pdf_call_count += 1
        self._pdf_total_cost += call_cost

        # ----- Try to extract chart_type from the JSON-ish response ----- #
        chart_type = self._extract_field(description, "chart_type")

        bbox = (
            float(image_rect.x0),
            float(image_rect.y0),
            float(image_rect.x1),
            float(image_rect.y1),
        )

        block = ChartBlock(
            description=description,
            chart_type=chart_type,
            page_num=page_num,
            bbox=bbox,
            extraction_cost_usd=round(call_cost, 6),
        )

        logger.info(
            "chart_extractor.chart_described",
            page_num=page_num,
            chart_type=chart_type,
            cost_usd=round(call_cost, 6),
            cumulative_cost_usd=round(self._pdf_total_cost, 6),
            call_count=self._pdf_call_count,
        )

        return block

    async def batch_extract(self, page: fitz.Page) -> list[ChartBlock]:
        """Extract descriptions for all images/charts on a page.

        Iterates over ``page.get_images()``, resolves each image's
        bounding rectangle, and concurrently sends them through
        ``extract_chart_description``.

        Args:
            page: A PyMuPDF ``fitz.Page`` object.

        Returns:
            List of ``ChartBlock`` objects (may be shorter than the
            number of images if limits are reached).
        """
        image_list = page.get_images(full=True)
        if not image_list:
            return []

        rects = self._resolve_image_rects(page, image_list)
        if not rects:
            return []

        # Create tasks but respect sequential ordering so the rate-limit
        # and cost counters stay accurate.  We use a semaphore to allow
        # moderate concurrency (3 at a time) while still incrementing
        # counters safely.
        sem = asyncio.Semaphore(3)

        async def _guarded(rect: fitz.Rect) -> ChartBlock | None:
            async with sem:
                return await self.extract_chart_description(page, rect)

        tasks = [_guarded(rect) for rect in rects]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        blocks: list[ChartBlock] = []
        for result in results:
            if isinstance(result, ChartBlock):
                blocks.append(result)
            elif isinstance(result, Exception):
                logger.error(
                    "chart_extractor.batch_task_error",
                    error=str(result),
                )

        logger.debug(
            "chart_extractor.batch_complete",
            page_num=page.number + 1,
            images_found=len(image_list),
            charts_extracted=len(blocks),
        )
        return blocks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_image_rects(
        page: fitz.Page,
        image_list: list,
    ) -> list[fitz.Rect]:
        """Map each embedded image to its bounding rectangle on the page.

        Falls back to the full page rect if the image reference cannot be
        located in the page's display list.

        Args:
            page: The PyMuPDF page.
            image_list: Output of ``page.get_images(full=True)``.

        Returns:
            List of ``fitz.Rect`` objects, one per image.
        """
        rects: list[fitz.Rect] = []

        for img_info in image_list:
            xref = img_info[0]
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rects.append(img_rects[0])
                else:
                    # Fallback: use the entire page
                    rects.append(page.rect)
            except Exception:
                logger.debug(
                    "chart_extractor.rect_fallback",
                    xref=xref,
                    page_num=page.number + 1,
                )
                rects.append(page.rect)

        return rects

    @staticmethod
    def _extract_field(text: str, field: str) -> str | None:
        """Best-effort extraction of a JSON field value from LLM output.

        Handles both quoted and unquoted values for robustness against
        slightly malformed JSON responses.

        Args:
            text: Raw LLM response text.
            field: The JSON key to look for.

        Returns:
            The extracted string value, or ``None`` if not found.
        """
        import re

        # Try "field": "value"
        pattern = rf'"{field}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # Try "field": value (unquoted / null)
        pattern_unquoted = rf'"{field}"\s*:\s*(\S+)'
        match_u = re.search(pattern_unquoted, text)
        if match_u:
            val = match_u.group(1).strip(",").strip()
            if val.lower() == "null":
                return None
            return val

        return None
