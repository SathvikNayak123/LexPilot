# LexPilot — AI Legal Intelligence for Indian Law

Multi-agent RAG system for Indian Supreme Court judgments with citation verification, knowledge graph traversal, and adversarial guardrails. Built on OpenAI Agents SDK with native agent handoffs.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Input Guardrail (injection / off-topic / advice block) │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
              LexPilot Orchestrator
           ┌──────────┬──────────┬──────────┐
           │          │          │          │
      Contract    Precedent  Compliance  Risk
      Analyst    Researcher   Auditor    Scorer
           │          │          │          │
           └──────────┴────┬─────┴──────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  GraphRAG Retrieval                                      │
│  Dense (SBERT) + Sparse (BM25) → RRF → Cross-Encoder   │
│  → Neo4j enrichment (authority / citation / overruled)  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Output Guardrail — 3-Tier Citation Verifier             │
│  Tier 1: EXISTS   Tier 2: NOT OVERRULED   Tier 3: NLI   │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

### Hybrid Retrieval
- **Dense search** via `all-mpnet-base-v2` sentence embeddings stored in Qdrant
- **Sparse search** via BM25 (rank-bm25, corpus-wide IDF, pickle-cached)
- **RRF fusion** (k=60) combining both ranked lists by position
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`, top-20 candidates)
- **Parent-child chunking** — child chunks indexed for precision, parent chunks fetched for LLM context
- Achieves **93% Recall@5** and **85% MMR** on 100-query benchmark

### Legal Knowledge Graph (Neo4j)
- Judgment nodes with typed citation edges: `CITES`, `OVERRULES`, `DISTINGUISHED_FROM`, `APPLIED`
- Edge types classified via zero-shot **InLegalBERT**, falls back to regex if model unavailable
- Graph re-ranker applies multiplicative boosts: court authority (SC > HC > District) × citation importance × overruled penalty (0.1×)
- Surfaces overruled-judgment warnings and citation-chain context at query time

### 3-Tier Citation Verification
Eliminates hallucinated citations before the response leaves the system:

| Tier | Check | Mechanism |
|------|-------|-----------|
| 1 | Citation exists | Lookup in `citation_index` with format normalization (11+ SCC/AIR/SCR variants) |
| 2 | Not overruled | `is_overruled` flag populated from Neo4j OVERRULES edges |
| 3 | Accurately characterized | NLI (entailment ≥ contradiction) + cosine similarity ≥ 0.50 |

- **79% citation accuracy** on adversarial benchmark (fabricated citations, overruled precedents, mischaracterization)
- Fabricated citations replaced inline; overruled citations flagged `[OVERRULED]`; mischaracterizations appended with correction

### DPDP Act 2023 Compliance Auditor
- Exhaustive map-reduce clause scanning via Instructor-validated structured outputs
- Rejects hallucinated section numbers via schema validation
- **74% compliance recall** on 12 annotated privacy policies

### Guardrails
- **Input**: Classifier agent (4-way: `legal_query` | `legal_advice_request` | `injection` | `off_topic`)
  - Injection and off-topic queries are hard-blocked
  - Legal advice requests are reframed as informational rather than blocked
- **Output**: Citation verifier tripwire — fabricated citations abort the response and trigger regeneration

---

## Project Structure

```
lexpilot_agents/          # Agent definitions + guardrails
  orchestrator.py         # Master agent with 4 specialist handoffs
  contract_agent.py       # Clause extraction & contract analysis
  precedent_agent.py      # Case law research & citation chain tracing
  compliance_agent.py     # DPDP Act 2023 auditing
  risk_agent.py           # Legal risk scoring
  guardrails/
    input_guardrails.py   # Injection / off-topic filter
    output_guardrails.py  # Citation verification tripwire

retrieval/
  hybrid_search.py        # Dense + Sparse → RRF → Rerank → Parent hydration
  qdrant_store.py         # Dual-vector Qdrant client (dense + sparse)
  bm25.py                 # BM25 sparse encoder (rank-bm25 wrapper)
  indexer.py              # Full indexing pipeline

knowledge_graph/
  graph_builder.py        # Builds Neo4j judgment graph from parsed docs
  graph_retriever.py      # GraphRAG: vector results + batched Neo4j enrichment
  neo4j_client.py         # Async Neo4j operations

citation/
  extractor.py            # Regex citation extractor (11 Indian reporter formats)
  verifier.py             # 3-tier verification with batched DB lookup

ingestion/
  pdf_parser.py           # PyMuPDF (text) + pdfplumber (tables) dual-engine parser
  document_processor.py   # Async orchestration: parse → LLM citation extraction
  models.py               # ParsedBlock, ParsedDocument pydantic models

evaluation/
  benchmark.py            # 50+ landmark SC cases: retrieval, QA, precision tests
  ragas_eval.py           # RAGAS suite: faithfulness, answer_relevancy, context_precision
  adversarial.py          # 25-case adversarial suite (injection, fabrication, overruled)

scripts/
  build_bm25_index.py     # Corpus-wide BM25 index builder (2-pass, O(corpus))
  populate_citation_metadata.py  # Sync Neo4j overruled status → Postgres citation_index
  diagnose_graphrag.py    # Cross-validate Neo4j ↔ Qdrant node counts and edge stats

api/routes/chat.py        # FastAPI SSE streaming endpoint
frontend/                 # Next.js UI with streaming chat + confidence banners
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Running instances of: **Qdrant**, **Neo4j**, **PostgreSQL**, **Redis**

### 1. Install dependencies
```bash
pip install -e ".[dev]"
cd frontend && npm install
```

### 2. Configure environment
```bash
cp .env.example .env
# Set: OPENROUTER_API_KEY, QDRANT_URL, NEO4J_URI, DATABASE_URL, REDIS_URL
```

### 3. Ingest documents
```bash
# Parse PDFs, chunk, embed, and index into Qdrant + Postgres
python -m ingestion.document_processor --dir data/judgments/

# Build BM25 sparse index from the indexed corpus
python scripts/build_bm25_index.py

# Build Neo4j knowledge graph
python scripts/build_semantic_graph.py

# Populate citation metadata (overruled status, holding summaries)
python scripts/populate_citation_metadata.py
```

### 4. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Start the frontend
```bash
cd frontend && npm run dev
```

---

## Evaluation

### Retrieval & QA benchmark
```bash
python -m evaluation.benchmark
# Measures Recall@5, MMR across 50+ landmark SC judgments
```

### RAGAS generation quality
```bash
python -m evaluation.ragas_eval
# Faithfulness, answer_relevancy, context_precision, context_recall
```

### Adversarial citation test suite
```bash
python -m evaluation.adversarial
# 25 test cases: prompt injection, fabricated citations, overruled precedents,
# legal advice seeking, off-topic queries
```

### Diagnose graph coverage
```bash
python scripts/diagnose_graphrag.py
# Cross-validates Neo4j node counts vs Qdrant, reports edge counts per type
```

---

## Benchmark Results

| Metric | Score | Notes |
|--------|-------|-------|
| Recall@5 | 93% | 100-query retrieval benchmark |
| MMR | 85% | Diversity-adjusted relevance |
| Citation accuracy | 79% | Adversarial fabrication benchmark |
| DPDP compliance recall | 74% | 12 annotated privacy policies |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent framework | OpenAI Agents SDK (native handoffs) |
| LLM routing | OpenRouter (Gemini 3 Flash) |
| Vector store | Qdrant (dual dense+sparse vectors) |
| Sparse retrieval | BM25 via rank-bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Knowledge graph | Neo4j 5.x |
| NLI (citation check) | cross-encoder/nli-deberta-v3-xsmall |
| Legal NER | InLegalBERT (typed relation extraction) |
| Database | PostgreSQL + SQLAlchemy async |
| Structured outputs | Instructor |
| API | FastAPI + SSE streaming |
| Frontend | Next.js 14 + Tailwind |
| Observability | structlog + Langfuse |
