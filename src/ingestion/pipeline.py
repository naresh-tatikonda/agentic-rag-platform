import os
from pathlib import Path
from dotenv import load_dotenv

# Import all pipeline modules built today
from edgar_downloader import get_cik, get_10k_urls, download_10k_text
from parser import parse_10k_html
from chunker import chunk_text
from embedder import get_embeddings_batch
from inserter import insert_chunks, verify_insertion

load_dotenv()


def run_pipeline(ticker: str, max_filings: int = 1):
    """
    Full end-to-end ingestion pipeline for a single ticker.

    Pipeline stages:
    1. EDGAR API  → fetch CIK and 10-K filing metadata
    2. Downloader → download raw 10-K HTML from SEC
    3. Parser     → strip HTML, extract clean text
    4. Chunker    → split into 512-token overlapping chunks
    5. Embedder   → generate 1536-dim vectors via OpenAI API
    6. Inserter   → bulk insert chunks + vectors into pgvector

    This is the production ingestion pattern:
    each stage is independently testable and replaceable.

    Args:
        ticker:       Stock ticker to ingest (e.g. 'AAPL')
        max_filings:  Number of annual 10-Ks to ingest (default 1)
    """
    print(f"\n{'='*50}")
    print(f"Starting ingestion pipeline for: {ticker}")
    print(f"{'='*50}")

    # Stage 1 — Resolve ticker to SEC CIK
    print(f"\n[1/6] Fetching CIK for {ticker}...")
    cik = get_cik(ticker)
    print(f"      CIK: {cik}")

    # Stage 2 — Get list of 10-K filings
    print(f"\n[2/6] Fetching 10-K filing list...")
    filings = get_10k_urls(cik, max_filings=max_filings)
    print(f"      Found {len(filings)} filing(s)")

    total_chunks_inserted = 0

    for filing in filings:
        accession  = filing["accession"]
        filed_date = filing["filed_date"]
        print(f"\n      Processing filing: {accession} ({filed_date})")

        # Stage 3 — Download raw HTML from EDGAR
        print(f"\n[3/6] Downloading 10-K HTML...")
        raw_html = download_10k_text(cik, accession)

        if not raw_html:
            print(f"      WARNING: No HTML retrieved for {accession}, skipping")
            continue

        print(f"      Downloaded {len(raw_html):,} characters")

        # Stage 4 — Parse and clean HTML into plain text
        print(f"\n[4/6] Parsing and cleaning HTML...")
        clean_text = parse_10k_html(raw_html)
        print(f"      Clean text: {len(clean_text):,} characters")

        # Stage 5 — Chunk into 512-token segments
        print(f"\n[5/6] Chunking text...")
        chunks = chunk_text(clean_text, ticker=ticker, filing_date=filed_date)
        print(f"      Generated {len(chunks)} chunks")

        # Stage 6 — Generate embedding vectors
        print(f"\n[6/6] Generating embedding vectors...")
        embedded_chunks = get_embeddings_batch(chunks, batch_size=100)
        print(f"      Embedded {len(embedded_chunks)} chunks")

        # Stage 7 — Bulk insert into pgvector
        print(f"\n[7/7] Inserting into pgvector...")
        inserted = insert_chunks(
            chunks=embedded_chunks,
            ticker=ticker,
            filing_type="10-K",
            filed_date=filed_date,
            cik=cik
        )
        print(f"      Inserted {inserted} chunks")
        total_chunks_inserted += inserted

    # Final verification
    print(f"\n{'='*50}")
    print(f"Pipeline complete for {ticker}")
    result = verify_insertion(ticker)
    print(f"Total chunks in DB:   {result['total_chunks']}")
    print(f"Chunks with vectors:  {result['embedded_chunks']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Run pipeline for AAPL — 1 filing to keep cost low for testing
    # Change max_filings=3 to ingest all 3 years
    run_pipeline(ticker="AAPL", max_filings=1)
