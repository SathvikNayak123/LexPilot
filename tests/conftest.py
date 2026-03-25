"""Shared pytest fixtures for the FinDocs test suite.

Provides reusable fixtures for chunks, evaluation datasets, mock LLM clients,
parsed documents, and temporary PDF files.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fpdf import FPDF

from processing.chunker import Chunk
from ingestion.models import (
    ChartBlock,
    ParsedBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)


# ---------------------------------------------------------------------------
# sample_chunks — 20 pre-built Chunk objects
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Return 20 pre-built Chunk objects with varied doc_types, chunk_levels, and content."""

    doc_types = [
        "rbi_circular",
        "sebi_factsheet",
        "nse_annual_report",
        "rbi_circular",
        "sebi_factsheet",
    ]
    chunk_types = ["text", "table", "chart", "mixed", "text"]

    chunks: list[Chunk] = []
    parent_ids: list[uuid.UUID] = []

    # 10 parent chunks
    for i in range(10):
        parent_id = uuid.uuid4()
        parent_ids.append(parent_id)
        chunks.append(
            Chunk(
                chunk_id=parent_id,
                parent_id=None,
                content=f"Parent chunk {i}: This is a sample financial document paragraph "
                f"discussing topic number {i}. It contains relevant information about "
                f"Indian financial markets and regulatory frameworks.",
                chunk_level="parent",
                chunk_type=chunk_types[i % len(chunk_types)],
                doc_source=f"/data/docs/doc_{i}.pdf",
                doc_type=doc_types[i % len(doc_types)],
                doc_date=f"2025-{(i % 12) + 1:02d}-15",
                page_num=(i % 5) + 1,
                headings_context=f"Section {i} > Subsection {i}.1",
                token_count=256 + i * 10,
                char_count=1024 + i * 40,
            )
        )

    # 10 child chunks (each referencing a parent)
    for i in range(10):
        chunks.append(
            Chunk(
                chunk_id=uuid.uuid4(),
                parent_id=parent_ids[i],
                content=f"Child chunk {i}: Detailed excerpt from parent chunk {i}. "
                f"Contains specific data points and metrics for analysis.",
                chunk_level="child",
                chunk_type=chunk_types[i % len(chunk_types)],
                doc_source=f"/data/docs/doc_{i}.pdf",
                doc_type=doc_types[i % len(doc_types)],
                doc_date=f"2025-{(i % 12) + 1:02d}-15",
                page_num=(i % 5) + 1,
                headings_context=f"Section {i} > Subsection {i}.1",
                token_count=64 + i * 5,
                char_count=256 + i * 20,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# sample_eval_dataset — 10 QA pairs
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_eval_dataset() -> list[dict[str, str]]:
    """Return 10 QA pairs as dicts with question, ground_truth_answer, doc_type, question_type."""

    return [
        {
            "question": "What is the current repo rate set by RBI?",
            "ground_truth_answer": "The current repo rate set by RBI is 6.50%.",
            "doc_type": "rbi_circular",
            "question_type": "factual",
        },
        {
            "question": "What was the NAV of SBI Bluechip Fund as of December 2025?",
            "ground_truth_answer": "The NAV of SBI Bluechip Fund was Rs 78.42 as of December 2025.",
            "doc_type": "sebi_factsheet",
            "question_type": "numerical",
        },
        {
            "question": "How has the SIP growth trend changed over the last 3 years?",
            "ground_truth_answer": "SIP inflows grew from Rs 13,000 crore/month in 2023 to Rs 21,000 crore/month in 2025, a 61.5% increase.",
            "doc_type": "sebi_factsheet",
            "question_type": "analytical",
        },
        {
            "question": "Compare expense ratios of Nifty 50 index funds from HDFC and ICICI.",
            "ground_truth_answer": "HDFC Nifty 50 Index Fund has an expense ratio of 0.20% while ICICI Prudential Nifty 50 Index Fund has 0.18%.",
            "doc_type": "sebi_factsheet",
            "question_type": "comparison",
        },
        {
            "question": "What are SEBI's new margin requirements for F&O trading?",
            "ground_truth_answer": "SEBI mandates a minimum margin of 50% for F&O positions, increased from 40%, effective January 2026.",
            "doc_type": "rbi_circular",
            "question_type": "regulatory",
        },
        {
            "question": "What is the total AUM of Indian mutual fund industry?",
            "ground_truth_answer": "The total AUM of the Indian mutual fund industry crossed Rs 65 lakh crore in December 2025.",
            "doc_type": "sebi_factsheet",
            "question_type": "numerical",
        },
        {
            "question": "What is the CRR maintained by banks as per RBI guidelines?",
            "ground_truth_answer": "Banks are required to maintain a CRR of 4.50% of their net demand and time liabilities.",
            "doc_type": "rbi_circular",
            "question_type": "factual",
        },
        {
            "question": "Summarize NSE's financial performance for FY2025.",
            "ground_truth_answer": "NSE reported revenue of Rs 14,780 crore and net profit of Rs 8,305 crore for FY2025, representing 22% YoY growth.",
            "doc_type": "nse_annual_report",
            "question_type": "analytical",
        },
        {
            "question": "What is the minimum lot size for Nifty 50 futures?",
            "ground_truth_answer": "The minimum lot size for Nifty 50 futures is 25 units.",
            "doc_type": "nse_annual_report",
            "question_type": "factual",
        },
        {
            "question": "Compare the 1-year returns of HDFC Mid-Cap and Kotak Mid-Cap funds.",
            "ground_truth_answer": "HDFC Mid-Cap Opportunities returned 32.5% while Kotak Emerging Equity returned 28.9% over the 1-year period ending Dec 2025.",
            "doc_type": "sebi_factsheet",
            "question_type": "comparison",
        },
    ]


# ---------------------------------------------------------------------------
# mock_llm — returns canned responses for any LLM call
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a mock LLM that returns canned responses for any call."""

    mock = MagicMock()

    # Sync call
    mock.generate.return_value = {
        "content": "The current repo rate is 6.50% as per the latest RBI circular.",
        "model": "mock-model-v1",
        "tokens_used": 150,
    }

    # Async call
    async_generate = AsyncMock(
        return_value={
            "content": "The current repo rate is 6.50% as per the latest RBI circular.",
            "model": "mock-model-v1",
            "tokens_used": 150,
        }
    )
    mock.agenerate = async_generate

    # Chat completions style
    chat_response = MagicMock()
    chat_response.choices = [
        MagicMock(
            message=MagicMock(content="The current repo rate is 6.50%.")
        )
    ]
    chat_response.usage = MagicMock(
        prompt_tokens=100, completion_tokens=50, total_tokens=150
    )
    chat_response.model = "mock-model-v1"
    mock.chat.completions.create = AsyncMock(return_value=chat_response)

    return mock


# ---------------------------------------------------------------------------
# mock_openai_client — Mock AsyncOpenAI with canned vision/chat responses
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Return a mock AsyncOpenAI client that returns canned vision and chat responses."""

    mock_client = MagicMock()

    # Build a realistic chat completion response
    choice = MagicMock()
    choice.message.content = (
        '{"chart_type": "bar", "title": "Revenue Growth FY2024-25", '
        '"axis_labels": {"x": "Quarter", "y": "Revenue (Cr)"}, '
        '"key_data_points": ["Q1: 3200", "Q2: 3450", "Q3: 3800", "Q4: 4100"], '
        '"summary": "The bar chart shows quarterly revenue growth from Rs 3,200 crore '
        'to Rs 4,100 crore over FY2024-25, a 28% increase."}'
    )

    usage = MagicMock()
    usage.prompt_tokens = 500
    usage.completion_tokens = 120
    usage.total_tokens = 620

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "gpt-4o"

    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    return mock_client


# ---------------------------------------------------------------------------
# mock_anthropic_client — Mock Anthropic client
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Return a mock Anthropic client that returns canned classification/judge responses."""

    mock_client = MagicMock()

    # Build a realistic Anthropic messages response
    content_block = MagicMock()
    content_block.text = (
        '{"query_type": "factual", "top_k_override": 5, '
        '"chunk_type_filter": "child", "doc_type_filter": "rbi_circular", '
        '"run_multi_search": false, "sub_queries": []}'
    )

    message_response = MagicMock()
    message_response.content = [content_block]
    message_response.model = "claude-3-haiku-20240307"
    message_response.usage = MagicMock(input_tokens=80, output_tokens=60)

    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=message_response)

    return mock_client


# ---------------------------------------------------------------------------
# sample_parsed_document — a ParsedDocument with mixed block types
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_parsed_document() -> ParsedDocument:
    """Return a ParsedDocument with mixed block types (text, table, chart)."""

    blocks = [
        ParsedBlock(
            content="## RBI Monetary Policy Update\nThe Reserve Bank of India has announced "
            "a 25 basis point cut in the repo rate, bringing it to 6.25%. This decision "
            "was taken during the February 2025 monetary policy meeting.",
            block_type="text",
            page_num=1,
            metadata={"source": "text_extractor"},
        ),
        ParsedBlock(
            content="The inflation target band remains at 2-6%, with the current CPI "
            "inflation at 5.22% for January 2025. Core inflation has moderated to 3.8%.",
            block_type="text",
            page_num=1,
            metadata={"source": "text_extractor"},
        ),
        ParsedBlock(
            content="| Metric | Value | Change |\n| --- | --- | --- |\n"
            "| Repo Rate | 6.25% | -25 bps |\n"
            "| Reverse Repo | 3.35% | No change |\n"
            "| CRR | 4.50% | No change |",
            block_type="table",
            page_num=2,
            metadata={
                "source": "table_extractor",
                "row_count": 3,
                "col_count": 3,
            },
        ),
        ParsedBlock(
            content="Bar chart showing repo rate trend from 2020 to 2025. "
            "Rates declined from 5.15% in 2020 to 4.00% in 2022, then "
            "increased to 6.50% in 2023, and decreased to 6.25% in 2025.",
            block_type="chart",
            page_num=3,
            metadata={
                "source": "chart_extractor",
                "chart_type": "bar",
                "extraction_cost_usd": 0.01,
            },
        ),
        ParsedBlock(
            content="The GDP growth projection for FY2025 has been revised upward to 7.2% "
            "from the earlier estimate of 7.0%. Agricultural output is expected to "
            "remain robust, supporting rural demand.",
            block_type="text",
            page_num=3,
            metadata={"source": "text_extractor"},
        ),
    ]

    return ParsedDocument(
        source_path=Path("/data/docs/rbi_policy_feb2025.pdf"),
        doc_type="rbi_circular",
        title="RBI Monetary Policy Statement February 2025",
        date=datetime(2025, 2, 7),
        blocks=blocks,
        total_pages=4,
        parsing_duration_seconds=2.45,
    )


# ---------------------------------------------------------------------------
# tmp_pdf — generate a simple test PDF with fpdf2
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    """Generate a simple test PDF with text using fpdf2 and return its path."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1 — heading + body text
    pdf.add_page()
    pdf.set_font("Helvetica", "B", size=18)
    pdf.cell(0, 10, text="RBI Monetary Policy Statement", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        7,
        text=(
            "The Reserve Bank of India has decided to keep the repo rate unchanged "
            "at 6.50 percent. The standing deposit facility rate remains at 6.25 percent "
            "and the marginal standing facility rate at 6.75 percent. The Bank Rate also "
            "remains unchanged at 6.75 percent. The MPC decided to remain focused on "
            "withdrawal of accommodation to ensure that inflation progressively aligns "
            "with the target."
        ),
    )
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", size=14)
    pdf.cell(0, 10, text="Economic Outlook", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        7,
        text=(
            "The real GDP growth for 2024-25 is projected at 7.0 percent with risks "
            "evenly balanced. CPI inflation for 2024-25 is projected at 5.4 percent. "
            "The current account deficit is expected to remain manageable."
        ),
    )

    # Page 2 — more body text
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        7,
        text=(
            "Global economic activity has remained resilient despite tight monetary "
            "conditions. Headline inflation in major economies is on a declining path "
            "but remains above targets. Financial markets have exhibited some volatility "
            "due to geopolitical uncertainties and shifting monetary policy expectations."
        ),
    )

    output_path = tmp_path / "test_document.pdf"
    pdf.output(str(output_path))
    return output_path
