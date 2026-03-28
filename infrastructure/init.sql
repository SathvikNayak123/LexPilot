CREATE EXTENSION IF NOT EXISTS vector;

-- Parent chunks (512-token context for LLM after retrieval)
CREATE TABLE IF NOT EXISTS parent_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_parent_doc ON parent_chunks(document_id);

-- Document metadata
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    doc_type TEXT NOT NULL,  -- "judgment", "contract", "statute", "policy"
    source TEXT,
    court TEXT,              -- For judgments: "Supreme Court", "Delhi HC", etc.
    date DATE,
    metadata JSONB DEFAULT '{}',
    parse_duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Verified citation index
CREATE TABLE IF NOT EXISTS citation_index (
    id SERIAL PRIMARY KEY,
    citation_string TEXT UNIQUE NOT NULL,  -- e.g., "AIR 2017 SC 4161"
    case_name TEXT NOT NULL,
    court TEXT NOT NULL,
    date DATE,
    holding_summary TEXT,
    is_overruled BOOLEAN DEFAULT FALSE,
    overruled_by TEXT,  -- Citation of overruling judgment
    subject_tags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_citation_string ON citation_index(citation_string);

-- Research sessions (cross-session memory)
CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    summary JSONB DEFAULT '{}',
    precedents_found TEXT[],
    clauses_analyzed JSONB DEFAULT '[]',
    compliance_findings JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Evaluation results
CREATE TABLE IF NOT EXISTS eval_results (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    score FLOAT NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User feedback
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    trace_id TEXT,
    rating INTEGER CHECK (rating IN (-1, 1)),
    comment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
