-- FinDocs: PostgreSQL initialization script
-- Creates pgvector extension and all application tables.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------------------------------------------------------------------------
-- chunks: stores document chunk embeddings and metadata produced by ingestion
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id    TEXT UNIQUE NOT NULL,
    parent_id   TEXT,
    content     TEXT NOT NULL,
    doc_type    TEXT,
    doc_date    DATE,
    chunk_type  TEXT,
    chunk_level INTEGER,
    headings_context TEXT,
    token_count INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id   ON chunks (chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id  ON chunks (parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type   ON chunks (doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_date   ON chunks (doc_date);

-- ---------------------------------------------------------------------------
-- eval_reports: evaluation run results written by the CI gate / eval harness
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_reports (
    id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id    TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    scores    JSONB NOT NULL DEFAULT '{}'::jsonb,
    passed    BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_eval_reports_run_id    ON eval_reports (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_reports_timestamp ON eval_reports (timestamp);

-- ---------------------------------------------------------------------------
-- user_feedback: explicit thumbs-up / thumbs-down and free-text feedback
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_feedback (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id   TEXT NOT NULL,
    session_id TEXT,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    score      INTEGER CHECK (score BETWEEN -1 AND 1),
    comment    TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_trace_id   ON user_feedback (trace_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_session_id ON user_feedback (session_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at ON user_feedback (created_at);

-- ---------------------------------------------------------------------------
-- drift_reports: periodic quality-drift detection snapshots
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_reports (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT now(),
    baseline_scores  JSONB NOT NULL DEFAULT '{}'::jsonb,
    current_scores   JSONB NOT NULL DEFAULT '{}'::jsonb,
    drift_pcts       JSONB NOT NULL DEFAULT '{}'::jsonb,
    alerts           JSONB NOT NULL DEFAULT '[]'::jsonb,
    triggered_retrain BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_timestamp ON drift_reports (timestamp);
