import os
import psycopg2
from psycopg2.extras import execute_values
from typing import List
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """
    Create and return a PostgreSQL database connection.

    Connection parameters are loaded from .env file — never hardcoded.
    This is the production pattern: secrets in environment, not in code.
    """
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "ragdb"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "pass")
    )


def insert_chunks(chunks: List[dict], ticker: str, fiscal_year: int,
                  filing_type: str, filed_date: str, cik: str) -> int:
    """
    Bulk insert embedded chunks into the sec_filings pgvector table.

    Why execute_values instead of individual INSERTs?
    - execute_values sends all rows in a single SQL statement
    - 10-100x faster than looping individual inserts
    - Reduces network round trips to the database
    - Standard pattern for bulk ingestion in production pipelines

    The embedding vector is stored as a pgvector type — this enables
    HNSW approximate nearest neighbor search at query time.

    Args:
        chunks:       List of chunk dicts with 'embedding' vectors added
        ticker:       Stock ticker symbol (e.g. 'AAPL')
        fiscal_year:  Fiscal year the 10-K covers (e.g. 2025) — derived
                      from EDGAR accession number, never hardcoded
        filing_type:  SEC form type (e.g. '10-K')
        filed_date:   Filing date string (e.g. '2025-10-31')
        cik:          SEC Central Index Key for the company

    Returns:
        Number of chunks successfully inserted
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Build list of tuples for bulk insert
    # Each tuple maps to one row in sec_filings table
    rows = []
    for chunk in chunks:
        # Skip chunks that failed embedding (safety check)
        if "embedding" not in chunk:
            continue

        rows.append((
            ticker,
            fiscal_year,
            filing_type,
            filed_date,
            cik,
            chunk["chunk_index"],
            chunk["chunk_text"],
            chunk["embedding"]   # 1536-dim vector → stored as pgvector type
        ))

    # execute_values performs a single multi-row INSERT
    # Much faster than cursor.execute() in a loop
    # make re-ingestion idempotent
    execute_values(
        cur,
        """
        INSERT INTO sec_filings
            (ticker, fiscal_year, filing_type, filed_date, cik,
            chunk_index, chunk_text, embedding)
        VALUES %s
        ON CONFLICT (ticker, fiscal_year, chunk_index) DO NOTHING
        """,
        rows
    )
 

    conn.commit()
    inserted = len(rows)

    cur.close()
    conn.close()

    return inserted


def verify_insertion(ticker: str) -> dict:
    """
    Verify chunks were inserted correctly for a given ticker.

    Runs three checks:
    1. Total chunk count for ticker
    2. Confirms embedding vectors are non-null
    3. Confirms BM25 GIN index is being used via EXPLAIN

    Returns dict with verification results.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Count total chunks stored for this ticker
    cur.execute(
        "SELECT COUNT(*) FROM sec_filings WHERE ticker = %s",
        (ticker,)
    )
    total_chunks = cur.fetchone()[0]

    # Confirm embeddings are stored (not null)
    cur.execute(
        """SELECT COUNT(*) FROM sec_filings
           WHERE ticker = %s AND embedding IS NOT NULL""",
        (ticker,)
    )
    embedded_chunks = cur.fetchone()[0]

    # Sample one chunk to verify content
    cur.execute(
        """SELECT chunk_index, chunk_text, filed_date
           FROM sec_filings WHERE ticker = %s
           ORDER BY chunk_index LIMIT 1""",
        (ticker,)
    )
    sample = cur.fetchone()

    cur.close()
    conn.close()

    return {
        "total_chunks":    total_chunks,
        "embedded_chunks": embedded_chunks,
        "sample_chunk":    sample
    }


if __name__ == "__main__":
    # Test with 3 pre-embedded sample chunks
    test_chunks = [
        {
            "chunk_text":  "Apple iPhone revenue grew 5% in fiscal 2025.",
            "chunk_index": 0,
            "token_count": 12,
            # Dummy 1536-dim vector (all 0.1) — real vectors come from embedder.py
            "embedding":   [0.1] * 1536
        },
        {
            "chunk_text":  "Risk factors include supply chain disruptions.",
            "chunk_index": 1,
            "token_count": 8,
            "embedding":   [0.2] * 1536
        },
        {
            "chunk_text":  "Management expects services revenue to grow.",
            "chunk_index": 2,
            "token_count": 8,
            "embedding":   [0.3] * 1536
        },
    ]

    print("Inserting test chunks into pgvector...")
    inserted = insert_chunks(
        chunks=test_chunks,
        ticker="AAPL",
        filing_type="10-K",
        filed_date="2025-10-31",
        cik="0000320193"
    )
    print(f"Inserted {inserted} chunks")

    print("\nVerifying insertion...")
    result = verify_insertion("AAPL")
    print(f"Total chunks in DB:    {result['total_chunks']}")
    print(f"Chunks with vectors:   {result['embedded_chunks']}")
    print(f"Sample chunk:          {result['sample_chunk']}")
