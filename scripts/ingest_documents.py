"""
Ingest all legal documents into the RAG pipeline.

Usage:
    python scripts/ingest_documents.py                      # Ingest all
    python scripts/ingest_documents.py --type judgment       # Judgments only
    python scripts/ingest_documents.py --type statute        # Statutes only
    python scripts/ingest_documents.py --file "path/to.pdf"  # Single file
    python scripts/ingest_documents.py --resume              # Skip already-ingested files
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path

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

# Court metadata for known judgments (helps citation verification & knowledge graph)
JUDGMENT_METADATA = {
    "Puttaswamy": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Kesavananda": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Vishaka": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Gurbaksh": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Shayara_Bano": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Maneka_Gandhi": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "M_C_Mehta": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Bachan_Singh": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Romesh_Thappar": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Arnesh_Kumar": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Vineeta_Sharma": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Additional_District": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "S_P_Gupta": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Cadila": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Pranay_Sethi": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Mcdermott": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Dashrath": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "Unni_Krishnan": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
    "D_Velusamy": {"court": "Supreme Court of India", "source": "indiankanoon.org"},
}


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(completed: set):
    PROGRESS_FILE.write_text(json.dumps(sorted(completed), indent=2))


def get_metadata(filename: str) -> dict:
    for key, meta in JUDGMENT_METADATA.items():
        if key.lower() in filename.lower():
            return meta
    return {"court": "Supreme Court of India", "source": "indiankanoon.org"}


def collect_pdfs(directory: Path) -> list[Path]:
    pdfs = sorted(directory.glob("*.pdf")) + sorted(directory.glob("*.PDF"))
    seen = set()
    unique = []
    for p in pdfs:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


async def ingest_file(pipeline: IndexingPipeline, pdf_path: Path, doc_type: str) -> dict:
    meta = get_metadata(pdf_path.name) if doc_type == "judgment" else {
        "source": "meity.gov.in" if "DPDP" in pdf_path.name else "indiacode.nic.in"
    }
    start = time.time()
    result = await pipeline.index_document(
        str(pdf_path),
        doc_type=doc_type,
        court=meta.get("court"),
        source=meta.get("source"),
    )
    elapsed = round(time.time() - start, 1)
    return {**result, "file": pdf_path.name, "time_s": elapsed}


async def ingest_directory(pipeline: IndexingPipeline, directory: Path, doc_type: str,
                           completed: set) -> list[dict]:
    all_pdfs = collect_pdfs(directory)
    pdfs = [p for p in all_pdfs if p.name not in completed]
    skipped = len(all_pdfs) - len(pdfs)

    if not all_pdfs:
        print(f"  No PDFs found in {directory}")
        return []

    if skipped:
        print(f"  Found {len(all_pdfs)} PDFs ({skipped} already done, {len(pdfs)} remaining)")
    else:
        print(f"  Found {len(pdfs)} PDFs in {directory.name}/")

    if not pdfs:
        print(f"  All files already ingested.")
        return []

    results = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i}/{len(pdfs)}] {pdf.name}...", end=" ", flush=True)
        try:
            result = await ingest_file(pipeline, pdf, doc_type)
            cites = result.get('citations', 0)
            print(f"OK ({result['parents']}P/{result['children']}C/{cites}cites, {result['time_s']}s)")
            results.append(result)
            completed.add(pdf.name)
            save_progress(completed)
        except Exception as e:
            print(f"FAILED: {e}")
            logger.error("ingestion_failed", path=str(pdf), error=str(e))
    return results


async def main():
    parser = argparse.ArgumentParser(description="Ingest legal documents into the RAG pipeline")
    parser.add_argument("--type", choices=["judgment", "statute", "all"], default="all",
                        help="Type of documents to ingest")
    parser.add_argument("--file", type=str, help="Ingest a single PDF file")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files that were already successfully ingested")
    args = parser.parse_args()

    pipeline = IndexingPipeline()
    completed = load_progress() if args.resume else set()

    print("=" * 60)
    print("LexPilot Document Ingestion")
    print(f"  Resume: {args.resume}")
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
        result = await ingest_file(pipeline, pdf, doc_type)
        cites = result.get('citations', 0)
        print(f"  OK ({result['parents']}P/{result['children']}C/{cites}cites, {result['time_s']}s)")
        all_results.append(result)
    else:
        if args.type in ("judgment", "all"):
            print(f"\n--- Judgments ({JUDGMENTS_DIR}) ---")
            results = await ingest_directory(pipeline, JUDGMENTS_DIR, "judgment", completed)
            all_results.extend(results)

        if args.type in ("statute", "all"):
            print(f"\n--- Statutes ({STATUTES_DIR}) ---")
            results = await ingest_directory(pipeline, STATUTES_DIR, "statute", completed)
            all_results.extend(results)

    total_time = round(time.time() - total_start, 1)
    total_parents = sum(r["parents"] for r in all_results)
    total_children = sum(r["children"] for r in all_results)

    print("\n" + "=" * 60)
    print(f"Ingestion Complete")
    print(f"  Documents: {len(all_results)}")
    print(f"  Parent chunks: {total_parents}")
    print(f"  Child chunks: {total_children}")
    print(f"  Total time: {total_time}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
