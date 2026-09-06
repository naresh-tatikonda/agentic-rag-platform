-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main documents table
CREATE TABLE IF NOT EXISTS sec_filings (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    filing_type VARCHAR(20) NOT NULL,
    accession   VARCHAR(25),          -- SEC accession number: unique per filing (10-K, each 10-Q, each 8-K)
    filed_date  DATE,
    cik         VARCHAR(20),
    chunk_index INTEGER,
    chunk_text  TEXT NOT NULL,
    embedding   vector(1536),
    created_at  TIMESTAMP DEFAULT NOW(),
    -- Keyed on accession, not (ticker, fiscal_year): a company files one 10-K
    -- but ~4 10-Qs and many 8-Ks per fiscal year, and their chunk_index all
    -- restart at 0 — keying on (ticker, fiscal_year, chunk_index) would make
    -- them clobber each other. accession is globally unique per filing.
    UNIQUE (accession, chunk_index)
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

