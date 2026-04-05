import requests
import json
import time
from pathlib import Path

# SEC requires a User-Agent header identifying your app + email
# Without this, SEC will block your requests (429 Too Many Requests)
HEADERS = {"User-Agent": "FinSightAI your@email.com"}

# Base URL for all SEC EDGAR API calls
BASE_URL = "https://data.sec.gov"


def get_cik(ticker: str) -> str:
    """
    Convert a stock ticker (e.g. 'AAPL') to a CIK number.
    CIK (Central Index Key) is SEC's unique ID for every company.
    We need CIK to look up filings — SEC doesn't use tickers directly.
    
    Example: AAPL → 0000320193
    """
    # SEC maintains a master JSON file mapping all tickers to CIK numbers
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(tickers_url, headers=HEADERS)
    data = r.json()

    # The JSON is a dict of dicts: {0: {ticker, cik_str, title}, 1: {...}, ...}
    # We loop through and match on ticker symbol (case-insensitive)
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            # CIK must be zero-padded to 10 digits for API calls
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker {ticker} not found in SEC database")


def get_10k_urls(cik: str, max_filings: int = 3) -> list:
    """
    Given a CIK, fetch the list of 10-K annual filings.
    Returns metadata (accession number, filing date) for each filing.
    
    Accession number is SEC's unique ID for each individual filing.
    Example: 0000320193-23-000106 (Apple's 2023 10-K)
    
    max_filings: how many 10-Ks to retrieve (3 = last 3 years)
    """
    # EDGAR submissions endpoint returns all filing history for a company
    url = f"{BASE_URL}/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS)
    data = r.json()

    # 'filings.recent' contains parallel arrays:
    # form[i], accessionNumber[i], filingDate[i] all correspond to same filing
    filings = data["filings"]["recent"]
    results = []

    for i, form in enumerate(filings["form"]):
        if form == "10-K" and len(results) < max_filings:
            accession = filings["accessionNumber"][i]  # e.g. 0000320193-23-000106
            filed_date = filings["filingDate"][i]       # e.g. 2023-11-03
            results.append({
                "cik": cik,
                "accession": accession,
                "filed_date": filed_date,
            })

    return results

def download_10k_text(cik: str, accession: str) -> str:
    """
    Download 10-K HTML using SEC EDGAR Archives.

    Key findings from debugging:
    - Index file is {accession}-index.htm (NOT -index.json)
    - Actual 10-K document filename follows pattern: {ticker}-{date}.htm
    - Base path: https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_fmt}/

    We parse the index HTM to extract the primary document filename,
    then download that document directly.
    """
    cik_int       = str(int(cik))
    accession_fmt = accession.replace("-", "")
    base_url      = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_fmt}"

    # Fetch the index page — lists all documents in this filing package
    index_url = f"{base_url}/{accession}-index.htm"
    print(f"      Fetching index: {index_url}")

    r = requests.get(index_url, headers=HEADERS)
    if r.status_code != 200:
        print(f"      Failed with status: {r.status_code}")
        return ""

    # Parse index HTML to find the primary 10-K document link
    # The primary document is typically ticker-date.htm (e.g. aapl-20250927.htm)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")

    doc_filename = None
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            # Column 4 in index table is the document type
            doc_type = cells[3].get_text(strip=True)
            if doc_type == "10-K":
                # Column 2 is the document filename link
                link = cells[2].find("a")
                if link and link["href"].endswith(".htm"):
                    doc_filename = link["href"].split("/")[-1]
                    break

    if not doc_filename:
        print("      No 10-K .htm document found in index")
        return ""

    # Download the actual 10-K document
    doc_url = f"{base_url}/{doc_filename}"
    print(f"      Downloading: {doc_url}")
    time.sleep(0.5)  # SEC rate limit — max 10 req/sec
    resp = requests.get(doc_url, headers=HEADERS)

    return resp.text

if __name__ == "__main__":
    # Create local directory to store raw downloaded filings
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    ticker = "AAPL"
    print(f"Fetching CIK for {ticker}...")
    cik = get_cik(ticker)
    print(f"CIK: {cik}")

    print("Fetching 10-K filing list...")
    filings = get_10k_urls(cik, max_filings=3)
    print(f"Found {len(filings)} filings: {filings}")
