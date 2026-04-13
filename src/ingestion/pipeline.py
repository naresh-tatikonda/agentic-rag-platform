# src/ingestion/pipeline.py
import os
from pathlib import Path
from dotenv import load_dotenv

from edgar_downloader import get_cik, get_10k_urls, download_10k_text
from parser import parse_10k_html
from chunker import chunk_text
from embedder import get_embeddings_batch
from inserter import insert_chunks, verify_insertion

load_dotenv()


def _extract_fiscal_year(filed_date: str, accession: str) -> int:
    """
    Derive fiscal_year from EDGAR metadata.

    Strategy (in priority order):
    1. Parse year from accession number  e.g. 0000320193-25-000123 → 2025
    2. Parse year from filed_date        e.g. "2025-11-03" → 2025
    3. Raise — never silently default to a hardcoded year

    Apple files its 10-K ~6 weeks after fiscal year end (Sep).
    Filing year == fiscal year for all annual 10-Ks.
    """
    # Strategy 1: accession number encodes 2-digit year in middle segment
    # Format: {cik}-{YY}-{sequence}  e.g. 0000320193-25-000123
    try:
        yy = int(accession.split("-")[1])          # "25" → 25
        century = 2000 if yy <= 99 else 1900
        fiscal_year = century + yy                 # 2025
        if 2000 <= fiscal_year <= 2099:            # sanity check
            return fiscal_year
    except (IndexError, ValueError):
        pass

    # Strategy 2: fall back to filed_date year
    try:
        fiscal_year = int(filed_date[:4])          # "2025-11-03" → 2025
        if 2000 <= fiscal_year <= 2099:
            return fiscal_year
    except (ValueError, TypeError):
        pass

    raise ValueError(
        f"Cannot derive fiscal_year from accession='{accession}' "
        f"filed_date='{filed_date}'. Pass --fiscal_year explicitly."
    )


def run_pipeline(ticker: str, max_filings: int = 1, fiscal_year_override: int = None):
    """
    Full end-to-end ingestion pipeline for a single ticker.

    Pipeline stages:
    1. EDGAR API  → fetch CIK and 10-K filing metadata
    2. Downloader → download raw 10-K HTML from SEC
    3. Parser     → strip HTML, extract clean text
    4. Chunker    → split into 512-token overlapping chunks
    5. Embedder   → generate 1536-dim vectors via OpenAI API
    6. Inserter   → bulk insert chunks + vectors into pgvector

    Args:
        ticker:               Stock ticker to ingest (e.g. 'AAPL')
        max_filings:          Number of annual 10-Ks to ingest (default 1)
        fiscal_year_override: Explicitly set fiscal_year (skips auto-detection).
                              Use only when EDGAR metadata is ambiguous.
    """
    print(f"\n{'='*50}")
    print(f"Starting ingestion pipeline for: {ticker}")
    print(f"{'='*50}")

    print(f"\n[1/6] Fetching CIK for {ticker}...")
    cik = get_cik(ticker)
    print(f"      CIK: {cik}")

    print(f"\n[2/6] Fetching 10-K filing list...")
    filings = get_10k_urls(cik, max_filings=max_filings)
    print(f"      Found {len(filings)} filing(s)")

    total_chunks_inserted = 0

    for filing in filings:
        accession  = filing["accession"]
        filed_date = filing["filed_date"]

        # ── Derive fiscal_year — NEVER hardcode or default silently ──────────
        if fiscal_year_override:
            fiscal_year = fiscal_year_override
            print(f"\n      fiscal_year={fiscal_year} (manual override)")
        else:
            fiscal_year = _extract_fiscal_year(filed_date, accession)
            print(f"\n      fiscal_year={fiscal_year} (auto-detected from accession)")

        print(f"\n      Processing: {accession} | filed: {filed_date} | FY: {fiscal_year}")

        print(f"\n[3/6] Downloading 10-K HTML...")
        raw_html = download_10k_text(cik, accession)
        if not raw_html:
            print(f"      WARNING: No HTML retrieved for {accession}, skipping")
            continue
        print(f"      Downloaded {len(raw_html):,} characters")

        print(f"\n[4/6] Parsing and cleaning HTML...")
        clean_text = parse_10k_html(raw_html)
        print(f"      Clean text: {len(clean_text):,} characters")

        print(f"\n[5/6] Chunking text...")
        chunks = chunk_text(clean_text, ticker=ticker, filing_date=filed_date)
        print(f"      Generated {len(chunks)} chunks")

        print(f"\n[6/6] Generating embedding vectors...")
        embedded_chunks = get_embeddings_batch(chunks, batch_size=100)
        print(f"      Embedded {len(embedded_chunks)} chunks")

        print(f"\n[7/7] Inserting into pgvector...")
        inserted = insert_chunks(
            chunks=embedded_chunks,
            ticker=ticker,
            fiscal_year=fiscal_year,       # ← FIXED: explicit, derived, never hardcoded
            filing_type="10-K",
            filed_date=filed_date,
            cik=cik
        )
        print(f"      Inserted {inserted} chunks for FY{fiscal_year}")
        total_chunks_inserted += inserted

    print(f"\n{'='*50}")
    print(f"Pipeline complete for {ticker}")
    result = verify_insertion(ticker)
    print(f"Total chunks in DB:   {result['total_chunks']}")
    print(f"Chunks with vectors:  {result['embedded_chunks']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEC 10-K ingestion pipeline")
    parser.add_argument("--ticker",       required=True,       help="Stock ticker e.g. AAPL")
    parser.add_argument("--max_filings",  type=int, default=1, help="Number of 10-Ks to ingest")
    parser.add_argument("--fiscal_year",  type=int, default=None,
                        help="Override fiscal year (auto-detected from EDGAR if omitted)")
    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        max_filings=args.max_filings,
        fiscal_year_override=args.fiscal_year,
    )