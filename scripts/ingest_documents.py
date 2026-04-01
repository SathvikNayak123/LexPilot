"""
Ingest all legal documents into the RAG pipeline.

Usage:
    python scripts/ingest_documents.py                      # Ingest all
    python scripts/ingest_documents.py --type judgment       # Judgments only
    python scripts/ingest_documents.py --type statute        # Statutes only
    python scripts/ingest_documents.py --file "path/to.pdf"  # Single file
    python scripts/ingest_documents.py --resume              # Skip already-ingested files
    python scripts/ingest_documents.py --concurrency 4       # Parallel workers (default: 4)

Metadata is loaded from data/judgment_metadata.yaml — edit that file to add or update
judgment entries without touching this script.
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.indexer import IndexingPipeline
import structlog

logger = structlog.get_logger()

# Document directories
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JUDGMENTS_DIR = DATA_DIR / "judgements"
STATUTES_DIR = DATA_DIR / "statutes"
PROGRESS_FILE = DATA_DIR / ".ingestion_progress.json"
METADATA_FILE = DATA_DIR / "judgment_metadata.yaml"


def load_judgment_metadata() -> dict:
    """Load judgment metadata from data/judgment_metadata.yaml.

    Falls back to an empty dict if the file is missing so ingestion still works
    (just without holding_summary / citation enrichment for known cases).
    """
    if not METADATA_FILE.exists():
        logger.warning("metadata_file_missing", path=str(METADATA_FILE))
        return {}
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    logger.info("metadata_loaded", entries=len(data), path=str(METADATA_FILE))
    return data


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(completed: set):
    PROGRESS_FILE.write_text(json.dumps(sorted(completed), indent=2))


def get_metadata(filename: str, judgment_metadata: dict) -> dict:
    """Match filename against metadata keys (case-insensitive substring match)."""
    lower = filename.lower()
    for key, meta in judgment_metadata.items():
        if key.lower() in lower:
            return meta
    return {"court": "Supreme Court of India", "source": "indiankanoon.org"}


def collect_pdfs(directory: Path) -> list[Path]:
    pdfs = sorted(directory.glob("*.pdf")) + sorted(directory.glob("*.PDF"))
    seen: set[str] = set()
    unique: list[Path] = []
    for p in pdfs:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


async def ingest_file(pipeline: IndexingPipeline, pdf_path: Path,
                      doc_type: str, judgment_metadata: dict) -> dict:
    meta = get_metadata(pdf_path.name, judgment_metadata) if doc_type == "judgment" else {
        "source": "meity.gov.in" if "DPDP" in pdf_path.name else "indiacode.nic.in"
    }
    start = time.time()
    result = await pipeline.index_document(
        str(pdf_path),
        doc_type=doc_type,
        court=meta.get("court"),
        source=meta.get("source"),
        citation=meta.get("citation"),
        holding_summary=meta.get("holding_summary"),
        is_overruled=meta.get("is_overruled", False),
        overruled_by=meta.get("overruled_by"),
    )
    elapsed = round(time.time() - start, 1)
    return {**result, "file": pdf_path.name, "time_s": elapsed}


async def ingest_directory(pipeline: IndexingPipeline, directory: Path,
                           doc_type: str, completed: set,
                           judgment_metadata: dict,
                           concurrency: int) -> list[dict]:
    all_pdfs = collect_pdfs(directory)
    pdfs = [p for p in all_pdfs if p.name not in completed]
    skipped = len(all_pdfs) - len(pdfs)

    if not all_pdfs:
        print(f"  No PDFs found in {directory}")
        return []

    if skipped:
        print(f"  Found {len(all_pdfs)} PDFs ({skipped} already done, {len(pdfs)} remaining)")
    else:
        print(f"  Found {len(pdfs)} PDFs in {directory.name}/  [concurrency={concurrency}]")

    if not pdfs:
        print("  All files already ingested.")
        return []

    semaphore = asyncio.Semaphore(concurrency)
    # Lock so concurrent tasks don't race on completed set + progress file
    progress_lock = asyncio.Lock()
    results: list[dict] = []

    async def _ingest_one(i: int, pdf: Path):
        async with semaphore:
            print(f"  [{i}/{len(pdfs)}] {pdf.name} ... starting")
            try:
                result = await ingest_file(pipeline, pdf, doc_type, judgment_metadata)
                print(f"  [{i}/{len(pdfs)}] {pdf.name} OK "
                      f"({result['parents']}P/{result['children']}C, {result['time_s']}s)")
                async with progress_lock:
                    results.append(result)
                    completed.add(pdf.name)
                    save_progress(completed)
            except Exception as e:
                print(f"  [{i}/{len(pdfs)}] {pdf.name} FAILED: {e}")
                logger.error("ingestion_failed", path=str(pdf), error=str(e))

    tasks = [_ingest_one(i, pdf) for i, pdf in enumerate(pdfs, 1)]
    await asyncio.gather(*tasks)
    return results


async def main():
    parser = argparse.ArgumentParser(description="Ingest legal documents into the RAG pipeline")
    parser.add_argument("--type", choices=["judgment", "statute", "all"], default="all",
                        help="Type of documents to ingest")
    parser.add_argument("--file", type=str, help="Ingest a single PDF file")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files that were already successfully ingested")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of documents to ingest in parallel (default: 4)")
    args = parser.parse_args()

    judgment_metadata = load_judgment_metadata()
    pipeline = IndexingPipeline()
    completed = load_progress() if args.resume else set()

    print("=" * 60)
    print("LexPilot Document Ingestion")
    print(f"  Resume: {args.resume}  |  Concurrency: {args.concurrency}")
    print("=" * 60)

    total_start = time.time()
    all_results = []

    if args.file:
        pdf = Path(args.file)
        if not pdf.exists():
            print(f"File not found: {pdf}")
            sys.exit(1)
        doc_type = "statute" if "statute" in str(pdf).lower() else "judgment"
        print(f"\nIngesting single file: {pdf.name} (type={doc_type})")
        result = await ingest_file(pipeline, pdf, doc_type, judgment_metadata)
        print(f"  OK ({result['parents']}P/{result['children']}C, {result['time_s']}s)")
        all_results.append(result)
    else:
        if args.type in ("judgment", "all"):
            print(f"\n--- Judgments ({JUDGMENTS_DIR}) ---")
            results = await ingest_directory(
                pipeline, JUDGMENTS_DIR, "judgment", completed, judgment_metadata, args.concurrency
            )
            all_results.extend(results)

        if args.type in ("statute", "all"):
            print(f"\n--- Statutes ({STATUTES_DIR}) ---")
            results = await ingest_directory(
                pipeline, STATUTES_DIR, "statute", completed, judgment_metadata, args.concurrency
            )
            all_results.extend(results)

    total_time = round(time.time() - total_start, 1)
    total_parents = sum(r["parents"] for r in all_results)
    total_children = sum(r["children"] for r in all_results)

    print("\n" + "=" * 60)
    print("Ingestion Complete")
    print(f"  Documents: {len(all_results)}")
    print(f"  Parent chunks: {total_parents}")
    print(f"  Child chunks: {total_children}")
    print(f"  Total time: {total_time}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
