# FinDocs – Financial Document Intelligence Pipeline

Production-grade LLMOps pipeline for Indian financial document intelligence, targeting retail investors.

## Features

- **Ingestion**: Scrapes RBI circulars, SEBI factsheets, NSE annual reports
- **Parsing**: Multimodal PDF extraction (text, tables, charts via vision LLM)
- **Chunking**: Semantic + parent-child strategy for optimal retrieval
- **Indexing**: Hybrid search (dense + sparse BM25) in Qdrant
- **Fine-tuning**: QLoRA fine-tuning of Phi-3-mini on synthetic financial QA
- **Evaluation**: RAGAS-based suite (5 metrics) running as CI quality gate
- **Serving**: FastAPI with prompt versioning via Langfuse
- **Monitoring**: Drift detection with automatic re-eval triggers

## Quick Start

```bash
cp .env.example .env
# Fill in API keys in .env

docker-compose up -d
pip install -e ".[dev]"
python -m api.main
```

## Evaluation

```bash
python -m evaluation.ci_gate --sample-size 50
```

## Development

```bash
ruff check .
mypy --strict .
pytest
```
