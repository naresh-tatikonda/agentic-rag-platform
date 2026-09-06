import time

import requests
from bs4 import BeautifulSoup

# SEC requires a User-Agent header identifying your app + email.
# Without this, SEC blocks requests (403 / 429).
HEADERS = {"User-Agent": "FinSightAI naresh.tde@gmail.com"}

BASE_URL = "https://data.sec.gov"

# Form types this pipeline knows how to ingest.
SUPPORTED_FORMS = ("10-K", "10-Q", "8-K")

_cik_cache: dict[str, str] = {}


def get_cik(ticker: str) -> str:
    """
    Convert a stock ticker (e.g. 'AAPL') to a zero-padded 10-digit CIK.
    SEC's API is keyed on CIK, not ticker. Result is cached per process.
    """
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    for entry in r.json().values():
        if entry["ticker"].upper() == ticker:
            cik = str(entry["cik_str"]).zfill(10)
            _cik_cache[ticker] = cik
            return cik

    raise ValueError(f"Ticker {ticker} not found in SEC database")


def get_filing_urls(cik: str, forms: list[str], max_per_form: dict[str, int]) -> list[dict]:
    """
    Return filing metadata for the requested form types, newest first,
    capped per form by max_per_form.

    The submissions endpoint's `filings.recent` holds ~1000 of the most
    recent filings — plenty for a 2-year window of 10-K/10-Q/8-K.

    Returns dicts: {cik, form, accession, filed_date}
    """
    url = f"{BASE_URL}/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]

    wanted = {f.upper() for f in forms}
    counts = {f.upper(): 0 for f in forms}
    results: list[dict] = []

    for i, form in enumerate(recent["form"]):
        form_u = form.upper()
        if form_u not in wanted:
            continue
        if counts[form_u] >= max_per_form.get(form_u, 0):
            continue
        counts[form_u] += 1
        results.append({
            "cik": cik,
            "form": form_u,
            "accession": recent["accessionNumber"][i],
            "filed_date": recent["filingDate"][i],
        })

    return results


def download_filing_text(cik: str, accession: str, form: str) -> str:
    """
    Download the primary HTML document for a filing.

    Strategy: fetch the filing index page, find the row whose type column
    matches `form`, download that document. If no exact type match (common
    for 8-Ks where the primary doc is typed oddly), fall back to the first
    non-index .htm document in the package.
    """
    cik_int = str(int(cik))
    accession_fmt = accession.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_fmt}"

    index_url = f"{base_url}/{accession}-index.htm"
    r = requests.get(index_url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"      index fetch failed ({r.status_code}): {index_url}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")

    exact_match = None
    first_htm = None
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        doc_type = cells[3].get_text(strip=True)
        link = cells[2].find("a")
        if not link or not link.get("href", "").endswith(".htm"):
            continue
        fname = link["href"].split("/")[-1]
        if fname.endswith("-index.htm"):
            continue
        if first_htm is None:
            first_htm = fname
        if doc_type.upper() == form.upper():
            exact_match = fname
            break

    doc_filename = exact_match or first_htm
    if not doc_filename:
        print(f"      no .htm document found in index for {accession}")
        return ""

    time.sleep(0.5)  # SEC rate limit: max 10 req/sec
    resp = requests.get(f"{base_url}/{doc_filename}", headers=HEADERS, timeout=60)
    return resp.text if resp.status_code == 200 else ""


if __name__ == "__main__":
    cik = get_cik("AAPL")
    print(f"AAPL CIK: {cik}")
    filings = get_filing_urls(cik, ["10-K", "10-Q", "8-K"], {"10-K": 2, "10-Q": 4, "8-K": 3})
    for f in filings:
        print(f"  {f['form']:6} {f['filed_date']}  {f['accession']}")
