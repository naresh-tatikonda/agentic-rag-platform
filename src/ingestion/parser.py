import re
from bs4 import BeautifulSoup

def parse_10k_html(raw_html: str) -> str:
    """
    Convert raw 10-K HTML into clean plain text.
    
    SEC 10-K filings are delivered as HTML with:
    - Lots of HTML tags (<p>, <table>, <div>, etc.)
    - XBRL inline data tags (financial metadata, not readable text)
    - Page headers, footers, exhibit markers
    - Excessive whitespace and special characters
    
    We strip all of that and return clean readable text only.
    """
    # BeautifulSoup parses the HTML document tree
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove XBRL inline tags — these are machine-readable financial
    # metadata embedded in the HTML, not human-readable text
    for tag in soup.find_all(["ix:nonfraction", "ix:nonnumeric"]):
        tag.decompose()  # completely remove the tag and its contents

    # Remove script and style blocks — JavaScript and CSS are not text
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Extract all remaining visible text from the HTML
    text = soup.get_text(separator=" ")

    # Remove SEC table of contents markers and page break artifacts
    # These appear as patterns like "Table of Contents" repeated throughout
    text = re.sub(r'Table of Contents', '', text, flags=re.IGNORECASE)

    # Collapse multiple whitespace characters (spaces, tabs, newlines)
    # into a single space for clean tokenization later
    text = re.sub(r'\s+', ' ', text)

    # Remove non-ASCII characters (some filings have unicode artifacts)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def extract_sections(text: str) -> dict:
    """
    Extract key sections from 10-K text for targeted chunking.
    
    A 10-K has standardized sections (Items) defined by SEC:
    - Item 1:  Business description
    - Item 1A: Risk Factors (most important for analysis)
    - Item 7:  Management Discussion & Analysis (MD&A)
    - Item 8:  Financial Statements
    
    Extracting by section lets us tag chunks with their source section,
    which improves retrieval relevance (e.g. only search Risk Factors).
    """
    sections = {}

    # Regex patterns for the major 10-K sections
    # (?i) = case insensitive, \s* = flexible whitespace
    patterns = {
        "business":      r"(?i)item\s*1[\.\s]+business",
        "risk_factors":  r"(?i)item\s*1a[\.\s]+risk\s*factors",
        "mda":           r"(?i)item\s*7[\.\s]+management",
        "financials":    r"(?i)item\s*8[\.\s]+financial\s*statements",
    }

    # Find start position of each section in the text
    positions = {}
    for section_name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            positions[section_name] = match.start()

    # Sort sections by their position in the document
    sorted_sections = sorted(positions.items(), key=lambda x: x[1])

    # Extract text between each section start and the next section start
    for i, (section_name, start_pos) in enumerate(sorted_sections):
        # End of this section = start of next section (or end of document)
        if i + 1 < len(sorted_sections):
            end_pos = sorted_sections[i + 1][1]
        else:
            end_pos = len(text)

        sections[section_name] = text[start_pos:end_pos].strip()

    return sections


if __name__ == "__main__":
    # Quick test with a dummy HTML snippet
    test_html = """
    <html><body>
    <p>Item 1. Business Apple Inc. designs and sells consumer electronics.</p>
    <p>Item 1A. Risk Factors Our business faces significant competition.</p>
    <p>Item 7. Management Discussion Revenue increased 5% year over year.</p>
    </body></html>
    """
    clean = parse_10k_html(test_html)
    print("Cleaned text:")
    print(clean[:300])

    sections = extract_sections(clean)
    print(f"\nSections found: {list(sections.keys())}")
    for name, content in sections.items():
        print(f"\n[{name}]: {content[:100]}...")
