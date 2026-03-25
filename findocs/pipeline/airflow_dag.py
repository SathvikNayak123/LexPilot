"""Airflow DAG for the FinDocs weekly data pipeline.

Runs every Monday at 02:00 IST.  The pipeline scrapes new documents from
RBI and SEBI/AMFI, parses PDFs, chunks and embeds the parsed content into
Qdrant, runs an evaluation check on a sample of questions, and alerts via
Slack if quality drift is detected.

Uses the TaskFlow API (``@task`` decorator style) for clean data passing
between tasks.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from airflow.decorators import dag, task

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default args shared by every task in the DAG
# ---------------------------------------------------------------------------

_DEFAULT_ARGS: dict[str, Any] = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
    "owner": "findocs",
}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------


@dag(
    dag_id="findocs_weekly_pipeline",
    schedule="0 2 * * 1",  # Every Monday at 2 AM IST
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["findocs", "production"],
    doc_md=__doc__,
    max_active_runs=1,
)
def findocs_pipeline() -> None:
    """FinDocs weekly ingestion, embedding, evaluation, and drift-detection pipeline."""

    # ------------------------------------------------------------------
    # Task 1a: Scrape RBI circulars
    # ------------------------------------------------------------------

    @task(task_id="scrape_rbi_circulars")
    def scrape_rbi_circulars() -> list[str]:
        """Run the RBI scraper and return a list of newly downloaded PDF paths.

        Uses ``asyncio.run`` because Airflow task callables are synchronous,
        but the underlying scraper is fully async.

        Returns:
            List of absolute file-path strings for every newly downloaded PDF.
        """
        from findocs.ingestion.scrapers.rbi_scraper import RBIScraper

        log = logger.bind(task="scrape_rbi_circulars")
        log.info("pipeline.scrape_rbi.start")

        async def _scrape() -> list[str]:
            scraper = RBIScraper(data_dir=Path("./data/raw/rbi"))
            docs = await scraper.scrape_circulars(
                year=datetime.now().year,
                max_docs=200,
            )
            return [str(doc.local_path.resolve()) for doc in docs]

        paths = asyncio.run(_scrape())
        log.info("pipeline.scrape_rbi.done", new_pdfs=len(paths))
        return paths

    # ------------------------------------------------------------------
    # Task 1b: Scrape SEBI / AMFI factsheets (parallel with 1a)
    # ------------------------------------------------------------------

    @task(task_id="scrape_sebi_factsheets")
    def scrape_sebi_factsheets() -> list[str]:
        """Run the SEBI/AMFI scraper and return newly downloaded PDF paths.

        Returns:
            List of absolute file-path strings for every newly downloaded PDF.
        """
        from findocs.ingestion.scrapers.sebi_scraper import SEBIScraper

        log = logger.bind(task="scrape_sebi_factsheets")
        log.info("pipeline.scrape_sebi.start")

        async def _scrape() -> list[str]:
            scraper = SEBIScraper(data_dir=Path("./data/raw/sebi"))
            docs = await scraper.scrape_factsheets(
                year=datetime.now().year,
                month=datetime.now().month,
                max_docs=200,
            )
            return [str(doc.local_path.resolve()) for doc in docs]

        paths = asyncio.run(_scrape())
        log.info("pipeline.scrape_sebi.done", new_pdfs=len(paths))
        return paths

    # ------------------------------------------------------------------
    # Task 2: Parse PDFs (depends on both scrapers)
    # ------------------------------------------------------------------

    @task(task_id="parse_documents")
    def parse_documents(pdf_paths: list[str]) -> list[str]:
        """Parse each downloaded PDF using PyMuPDF text + table extractors.

        Extracts text blocks and tables from every page of every PDF and
        serialises the resulting ``ParsedDocument`` objects to JSON files
        inside ``./data/parsed/``.

        Args:
            pdf_paths: Absolute paths to PDF files to parse.

        Returns:
            List of document IDs (stem of the parsed JSON file) for
            downstream consumption.
        """
        import time

        import fitz

        from findocs.ingestion.models import ParsedBlock, ParsedDocument
        from findocs.ingestion.parsers.table_extractor import TableExtractor
        from findocs.ingestion.parsers.text_extractor import TextExtractor

        log = logger.bind(task="parse_documents", total_pdfs=len(pdf_paths))
        log.info("pipeline.parse.start")

        text_extractor = TextExtractor()
        table_extractor = TableExtractor()
        parsed_dir = Path("./data/parsed")
        parsed_dir.mkdir(parents=True, exist_ok=True)

        doc_ids: list[str] = []

        for pdf_path_str in pdf_paths:
            pdf_path = Path(pdf_path_str)
            if not pdf_path.exists():
                log.warning("pipeline.parse.file_not_found", path=pdf_path_str)
                continue

            t0 = time.monotonic()
            blocks: list[ParsedBlock] = []

            try:
                pdf_doc = fitz.open(str(pdf_path))
                total_pages = len(pdf_doc)

                for page_idx in range(total_pages):
                    page = pdf_doc[page_idx]
                    page_num = page_idx + 1

                    # Extract text blocks
                    text_blocks = text_extractor.extract_blocks(page)
                    for tb in text_blocks:
                        blocks.append(
                            ParsedBlock(
                                content=tb.content,
                                block_type="text",
                                page_num=page_num,
                                metadata={
                                    "is_heading": tb.is_heading,
                                    "font_size": tb.font_size,
                                },
                            )
                        )

                    # Extract tables
                    table_blocks = table_extractor.extract_tables(pdf_path, page_num)
                    for tbl in table_blocks:
                        blocks.append(
                            ParsedBlock(
                                content=tbl.markdown_content,
                                block_type="table",
                                page_num=page_num,
                                metadata={
                                    "caption": tbl.caption,
                                    "row_count": tbl.row_count,
                                    "col_count": tbl.col_count,
                                    "numerical_summary": tbl.numerical_summary,
                                },
                            )
                        )

                pdf_doc.close()
                duration = time.monotonic() - t0

                # Determine doc_type from directory structure
                doc_type: str = "rbi_circular"
                if "sebi" in pdf_path_str.lower() or "amfi" in pdf_path_str.lower():
                    doc_type = "sebi_factsheet"

                parsed_doc = ParsedDocument(
                    source_path=pdf_path,
                    doc_type=doc_type,  # type: ignore[arg-type]
                    title=pdf_path.stem,
                    date=None,
                    blocks=blocks,
                    total_pages=total_pages,
                    parsing_duration_seconds=round(duration, 3),
                )

                # Persist parsed document as JSON
                doc_id = pdf_path.stem
                out_path = parsed_dir / f"{doc_id}.json"
                out_path.write_text(
                    parsed_doc.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                doc_ids.append(doc_id)

                log.info(
                    "pipeline.parse.doc_done",
                    doc_id=doc_id,
                    blocks=len(blocks),
                    pages=total_pages,
                    duration_s=round(duration, 3),
                )

            except Exception:
                log.exception("pipeline.parse.doc_error", path=pdf_path_str)

        log.info("pipeline.parse.done", parsed_count=len(doc_ids))
        return doc_ids

    # ------------------------------------------------------------------
    # Task 3: Chunk, embed, and upsert to Qdrant
    # ------------------------------------------------------------------

    @task(task_id="chunk_and_embed")
    def chunk_and_embed(doc_ids: list[str]) -> int:
        """Semantically chunk parsed documents, embed, and upsert to Qdrant.

        For each parsed document JSON:
          1. Load the ``ParsedDocument`` from disk.
          2. Run ``SemanticChunker.chunk_document()`` to produce parent/child
             chunks.
          3. Generate dense embeddings via ``EmbeddingService``.
          4. Generate BM25 sparse vectors via ``BM25Indexer``.
          5. Upsert child chunks (with both vectors) to Qdrant.

        Args:
            doc_ids: Document IDs (filenames without extension) output by
                the parse step.

        Returns:
            Total number of chunks upserted across all documents.
        """
        from sentence_transformers import SentenceTransformer

        from findocs.config.config import get_settings
        from findocs.ingestion.models import ParsedDocument
        from findocs.processing.bm25_indexer import BM25Indexer
        from findocs.processing.chunker import SemanticChunker
        from findocs.processing.embedder import EmbeddingService
        from findocs.retrieval.qdrant_client import (
            Chunk as QdrantChunk,
            FinDocsQdrantClient,
        )

        log = logger.bind(task="chunk_and_embed", doc_count=len(doc_ids))
        log.info("pipeline.chunk_embed.start")

        settings = get_settings()
        parsed_dir = Path("./data/parsed")

        # Initialise shared services
        embedding_service = EmbeddingService()
        st_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        chunker = SemanticChunker(
            embedding_model=st_model,
            parent_max_tokens=settings.CHUNK_SIZE_TOKENS,
            child_max_tokens=settings.CHILD_CHUNK_SIZE_TOKENS,
            child_overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        )

        total_chunks = 0

        async def _upsert_all() -> int:
            qdrant = FinDocsQdrantClient()
            await qdrant.create_collection()
            upserted = 0

            for doc_id in doc_ids:
                json_path = parsed_dir / f"{doc_id}.json"
                if not json_path.exists():
                    log.warning("pipeline.chunk_embed.missing_json", doc_id=doc_id)
                    continue

                raw_json = json_path.read_text(encoding="utf-8")
                parsed_doc = ParsedDocument.model_validate_json(raw_json)

                # Chunk
                chunks = chunker.chunk_document(parsed_doc)
                if not chunks:
                    log.warning("pipeline.chunk_embed.no_chunks", doc_id=doc_id)
                    continue

                # Prepare child chunks for Qdrant
                child_chunks = [c for c in chunks if c.chunk_level == "child"]
                if not child_chunks:
                    log.warning("pipeline.chunk_embed.no_children", doc_id=doc_id)
                    continue

                child_texts = [c.content for c in child_chunks]

                # Dense embeddings
                dense_vectors = embedding_service.embed_texts(child_texts)

                # BM25 sparse vectors
                bm25 = BM25Indexer(corpus=child_texts)
                sparse_vectors = bm25.batch_get_sparse_vectors(child_texts)

                # Map chunker Chunks -> Qdrant Chunks
                qdrant_chunks = [
                    QdrantChunk(
                        chunk_id=str(c.chunk_id),
                        content=c.content,
                        parent_id=str(c.parent_id) if c.parent_id else None,
                        doc_type=c.doc_type or parsed_doc.doc_type,
                        doc_date=c.doc_date or (
                            parsed_doc.date.isoformat() if parsed_doc.date else None
                        ),
                        page_num=c.page_num,
                        chunk_type=c.chunk_level,
                    )
                    for c in child_chunks
                ]

                await qdrant.upsert_chunks(qdrant_chunks, dense_vectors, sparse_vectors)
                upserted += len(qdrant_chunks)

                log.info(
                    "pipeline.chunk_embed.doc_done",
                    doc_id=doc_id,
                    parent_chunks=sum(1 for c in chunks if c.chunk_level == "parent"),
                    child_chunks=len(child_chunks),
                )

            return upserted

        total_chunks = asyncio.run(_upsert_all())
        log.info("pipeline.chunk_embed.done", total_chunks=total_chunks)
        return total_chunks

    # ------------------------------------------------------------------
    # Task 4: Run evaluation check
    # ------------------------------------------------------------------

    @task(task_id="run_eval_check")
    def run_eval_check(chunk_count: int) -> dict[str, float]:
        """Run a RAGAS evaluation suite on a sample of questions.

        Loads a 20-question evaluation dataset, runs each question through
        the RAG pipeline, and computes faithfulness, context precision, and
        answer relevance metrics.

        Args:
            chunk_count: Number of chunks upserted (used for logging; also
                serves as a data-dependency to ensure this task runs after
                embedding).

        Returns:
            Dictionary mapping metric names to their float scores, e.g.
            ``{"faithfulness": 0.88, "context_precision": 0.82, ...}``.
        """
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        from findocs.config.config import get_settings
        from findocs.processing.embedder import EmbeddingService
        from findocs.retrieval.qdrant_client import FinDocsQdrantClient

        log = logger.bind(task="run_eval_check", chunk_count=chunk_count)
        log.info("pipeline.eval.start")

        settings = get_settings()
        eval_sample_size = min(settings.WEEKLY_EVAL_QUESTION_SAMPLE, 20)

        # Load evaluation dataset from disk
        eval_path = Path("./data/eval/eval_questions.json")
        if not eval_path.exists():
            log.warning("pipeline.eval.no_eval_dataset", path=str(eval_path))
            return {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "answer_relevance": 0.0,
            }

        raw_eval = json.loads(eval_path.read_text(encoding="utf-8"))
        eval_questions: list[dict[str, Any]] = raw_eval[:eval_sample_size]

        if not eval_questions:
            log.warning("pipeline.eval.empty_dataset")
            return {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "answer_relevance": 0.0,
            }

        # Run each question through the retrieval pipeline to gather
        # contexts and answers
        embedding_service = EmbeddingService()

        async def _retrieve_contexts() -> tuple[list[list[str]], list[str]]:
            qdrant = FinDocsQdrantClient()
            all_contexts: list[list[str]] = []
            all_answers: list[str] = []

            for item in eval_questions:
                question = item["question"]
                query_vec = embedding_service.embed_query(question)
                results = await qdrant.search_dense(query_vec, top_k=5)
                contexts = [r.content for r in results]
                all_contexts.append(contexts)
                # Use the ground-truth answer if available, otherwise use
                # the top context as a proxy answer for metric calculation
                answer = item.get("answer", contexts[0] if contexts else "")
                all_answers.append(answer)

            return all_contexts, all_answers

        contexts, answers = asyncio.run(_retrieve_contexts())

        # Build RAGAS evaluation dataset
        eval_data = {
            "question": [q["question"] for q in eval_questions],
            "answer": answers,
            "contexts": contexts,
            "ground_truth": [
                q.get("ground_truth", q.get("answer", "")) for q in eval_questions
            ],
        }
        eval_dataset = Dataset.from_dict(eval_data)

        # Run RAGAS evaluation
        try:
            result = evaluate(
                eval_dataset,
                metrics=[faithfulness, context_precision, answer_relevancy],
            )
            scores: dict[str, float] = {
                "faithfulness": round(float(result.get("faithfulness", 0.0)), 4),
                "context_precision": round(
                    float(result.get("context_precision", 0.0)), 4
                ),
                "answer_relevance": round(
                    float(result.get("answer_relevancy", 0.0)), 4
                ),
            }
        except Exception:
            log.exception("pipeline.eval.ragas_error")
            scores = {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "answer_relevance": 0.0,
            }

        log.info("pipeline.eval.done", scores=scores)
        return scores

    # ------------------------------------------------------------------
    # Task 5: Check for quality drift
    # ------------------------------------------------------------------

    @task(task_id="check_drift")
    def check_drift(eval_scores: dict[str, float]) -> bool:
        """Compare current evaluation scores against configured thresholds.

        Drift is detected when *any* metric falls below its minimum
        acceptable threshold defined in project settings.

        Args:
            eval_scores: Metric name-to-score mapping from the eval step.

        Returns:
            ``True`` if quality drift is detected, ``False`` otherwise.
        """
        from findocs.config.config import get_settings

        log = logger.bind(task="check_drift")
        log.info("pipeline.drift.start", scores=eval_scores)

        settings = get_settings()

        thresholds: dict[str, float] = {
            "faithfulness": settings.MIN_FAITHFULNESS,
            "context_precision": settings.MIN_CONTEXT_PRECISION,
            "answer_relevance": settings.MIN_ANSWER_RELEVANCE,
        }

        drift_detected = False
        violations: list[dict[str, Any]] = []

        for metric, threshold in thresholds.items():
            current = eval_scores.get(metric, 0.0)
            if current < threshold:
                drift_detected = True
                violations.append(
                    {
                        "metric": metric,
                        "current": current,
                        "threshold": threshold,
                        "gap": round(threshold - current, 4),
                    }
                )

        if drift_detected:
            log.warning(
                "pipeline.drift.detected",
                violations=violations,
            )
        else:
            log.info("pipeline.drift.none", scores=eval_scores)

        return drift_detected

    # ------------------------------------------------------------------
    # Task 6: Send Slack alert if drift was detected
    # ------------------------------------------------------------------

    @task(task_id="alert_if_drift")
    def alert_if_drift(drift_detected: bool) -> None:
        """Send a Slack notification when quality drift is detected.

        Posts a message to the configured Slack webhook URL containing the
        drift details.  If no drift is detected, this task is a no-op.

        Args:
            drift_detected: Boolean flag from the drift-check step.
        """
        import httpx

        from findocs.config.config import get_settings

        log = logger.bind(task="alert_if_drift", drift_detected=drift_detected)

        if not drift_detected:
            log.info("pipeline.alert.skipped", reason="no_drift")
            return

        settings = get_settings()
        webhook_url = settings.SLACK_WEBHOOK_URL

        if not webhook_url:
            log.warning("pipeline.alert.no_webhook_url")
            return

        message = {
            "text": (
                ":warning: *FinDocs Quality Drift Detected*\n"
                "The weekly evaluation pipeline detected that one or more "
                "RAG quality metrics have fallen below their configured "
                "thresholds.\n\n"
                f"*Thresholds:*\n"
                f"  - Faithfulness >= {settings.MIN_FAITHFULNESS}\n"
                f"  - Context Precision >= {settings.MIN_CONTEXT_PRECISION}\n"
                f"  - Answer Relevance >= {settings.MIN_ANSWER_RELEVANCE}\n\n"
                "Please review the latest evaluation run in MLflow and "
                "investigate the root cause."
            ),
        }

        log.info("pipeline.alert.sending")

        try:
            response = httpx.post(
                webhook_url,
                json=message,
                timeout=30.0,
            )
            response.raise_for_status()
            log.info("pipeline.alert.sent", status=response.status_code)
        except (httpx.HTTPStatusError, httpx.TransportError):
            log.exception("pipeline.alert.failed")

    # ------------------------------------------------------------------
    # DAG topology: scrape in parallel -> merge -> parse -> chunk -> eval -> drift -> alert
    # ------------------------------------------------------------------

    rbi_paths = scrape_rbi_circulars()
    sebi_paths = scrape_sebi_factsheets()

    # Merge the two path lists before parsing
    @task(task_id="merge_pdf_paths")
    def merge_pdf_paths(rbi: list[str], sebi: list[str]) -> list[str]:
        """Combine PDF path lists from parallel scraping tasks.

        Args:
            rbi: Paths from the RBI scraper.
            sebi: Paths from the SEBI scraper.

        Returns:
            Combined deduplicated list of PDF paths.
        """
        combined = list(dict.fromkeys(rbi + sebi))  # preserves order, deduplicates
        logger.info(
            "pipeline.merge_paths",
            rbi_count=len(rbi),
            sebi_count=len(sebi),
            total=len(combined),
        )
        return combined

    all_paths = merge_pdf_paths(rbi=rbi_paths, sebi=sebi_paths)
    doc_ids = parse_documents(pdf_paths=all_paths)
    chunk_count = chunk_and_embed(doc_ids=doc_ids)
    eval_scores = run_eval_check(chunk_count=chunk_count)
    drift_flag = check_drift(eval_scores=eval_scores)
    alert_if_drift(drift_detected=drift_flag)


# Instantiate the DAG so Airflow can discover it
findocs_weekly_dag = findocs_pipeline()
