# src/ingestion/pipeline.py
"""
End-to-end SEC filing ingestion pipeline.

Per ticker, per form type:
    EDGAR API  -> filing list (capped per form)
    Downloader -> primary HTML document
    Parser     -> clean plain text (form-agnostic)
    Chunker    -> 512-token overlapping chunks
    Embedder   -> 1536-dim vectors (text-embedding-3-small)
    Inserter   -> bulk upsert into pgvector (idempotent on accession)

Usage:
    python pipeline.py --tickers AVGO,AMD,TSLA,AAPL \
                       --forms 10-K,10-Q,8-K \
                       --counts 10-K=2,10-Q=8,8-K=6
"""

import argparse
import os

from dotenv import load_dotenv

from edgar_downloader import get_cik, get_filing_urls, download_filing_text, SUPPORTED_FORMS
from parser import parse_filing_html
from chunker import chunk_text
from embedder import get_embeddings_batch
from inserter import insert_chunks, verify_insertion

load_dotenv()

# 8-Ks whose cleaned text is shorter than this are cover-page-only filings
# (an exhibit pointer, a one-line Item 5.02 note) — no retrievable content.
MIN_FILING_CHARS = 800


def _extract_fiscal_year(filed_date: str, accession: str) -> int:
    """
    Derive fiscal_year from EDGAR metadata (accession year, then filed_date year).
    Approximate for 10-Q/8-K (uses the filing's calendar year, not its fiscal
    period) — good enough for the ticker+year retrieval filter. Never a hardcoded
    default: raises if it can't be derived.
    """
    try:
        yy = int(accession.split("-")[1])
        fy = (2000 if yy <= 99 else 1900) + yy
        if 2000 <= fy <= 2099:
            return fy
    except (IndexError, ValueError):
        pass
    try:
        fy = int(filed_date[:4])
        if 2000 <= fy <= 2099:
            return fy
    except (ValueError, TypeError):
        pass
    raise ValueError(f"cannot derive fiscal_year from accession={accession!r} filed_date={filed_date!r}")


def ingest_ticker(ticker: str, forms: list[str], counts: dict[str, int]) -> int:
    print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
    cik = get_cik(ticker)
    print(f"  CIK: {cik}")

    filings = get_filing_urls(cik, forms, counts)
    print(f"  {len(filings)} filing(s) to process")

    total = 0
    for f in filings:
        accession, form, filed_date = f["accession"], f["form"], f["filed_date"]
        try:
            fiscal_year = _extract_fiscal_year(filed_date, accession)
        except ValueError as e:
            print(f"  SKIP {form} {accession}: {e}")
            continue

        print(f"\n  [{form}] {accession}  filed {filed_date}  FY{fiscal_year}")

        raw = download_filing_text(cik, accession, form)
        if not raw:
            print("    no document retrieved, skipping")
            continue

        text = parse_filing_html(raw)
        if len(text) < MIN_FILING_CHARS:
            print(f"    only {len(text)} chars of text, skipping (cover-page-only filing)")
            continue

        chunks = chunk_text(text, ticker=ticker, filing_date=filed_date)
        print(f"    {len(text):,} chars -> {len(chunks)} chunks")
        if not chunks:
            continue

        embedded = get_embeddings_batch(chunks, batch_size=100)
        inserted = insert_chunks(
            chunks=embedded,
            ticker=ticker,
            fiscal_year=fiscal_year,
            filing_type=form,
            filed_date=filed_date,
            cik=cik,
            accession=accession,
        )
        print(f"    inserted {inserted} chunks")
        total += inserted

    result = verify_insertion(ticker)
    print(f"\n  {ticker} total in DB: {result['total_chunks']} chunks ({result['embedded_chunks']} embedded)")
    return total


def _parse_counts(spec: str, forms: list[str]) -> dict[str, int]:
    """'10-K=2,10-Q=8,8-K=6' -> {'10-K': 2, ...}; missing forms default to 0."""
    counts = {f: 0 for f in forms}
    if spec:
        for pair in spec.split(","):
            k, _, v = pair.partition("=")
            counts[k.strip().upper()] = int(v)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SEC filing ingestion pipeline")
    ap.add_argument("--tickers", required=True, help="Comma-separated, e.g. AVGO,AMD,TSLA,AAPL")
    ap.add_argument("--forms", default="10-K", help=f"Comma-separated from {SUPPORTED_FORMS}")
    ap.add_argument("--counts", default="", help="Per-form caps, e.g. 10-K=2,10-Q=8,8-K=6")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    forms = [f.strip().upper() for f in args.forms.split(",") if f.strip()]
    bad = [f for f in forms if f not in SUPPORTED_FORMS]
    if bad:
        raise SystemExit(f"unsupported forms: {bad} (supported: {SUPPORTED_FORMS})")
    counts = _parse_counts(args.counts, forms)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    grand_total = 0
    for t in tickers:
        grand_total += ingest_ticker(t, forms, counts)
    print(f"\n{'=' * 60}\nDone. {grand_total} chunks inserted across {len(tickers)} ticker(s).")
