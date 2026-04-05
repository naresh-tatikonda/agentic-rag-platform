-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main documents table
CREATE TABLE IF NOT EXISTS sec_filings (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(10) NOT NULL,
    filing_type VARCHAR(20) NOT NULL,
    filed_date  DATE,
    cik         VARCHAR(20),
    chunk_index INTEGER,
    chunk_text  TEXT NOT NULL,
    embedding   vector(1536),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- HNSW index for fast ANN search
CREATE INDEX IF NOT EXISTS sec_filings_embedding_idx
ON sec_filings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- BM25 full-text search index
CREATE INDEX IF NOT EXISTS sec_filings_fts_idx
ON sec_filings
USING gin(to_tsvector('english', chunk_text));

