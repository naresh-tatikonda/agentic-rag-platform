"""
agents/nodes/query_analyzer.py
-------------------------------
QueryAnalyzer Node — the first node in the LangGraph pipeline.

Responsibility:
    Takes the raw user query and uses an LLM to extract:
    - ticker      : stock symbol (e.g. "AAPL")
    - fiscal_year : fiscal year the 10-K covers (e.g. 2023)
    - intent      : query intent (e.g. "risk_analysis")

Why an LLM for extraction?
    User queries are messy: "What risks did Apple face last year?"
    A simple regex won't handle "Apple" → "AAPL" or "last year" → 2023.
    An LLM handles this naturally via structured JSON output.

Output:
    Updates AgentState with: ticker, fiscal_year, intent
"""

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AgentState

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── LLM client (gpt-4o-mini for cost efficiency on extraction tasks) ──────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,          # Deterministic output — we want consistent extraction
    max_tokens=200,         # Extraction only needs a short JSON response
)

# ── Supported intents ─────────────────────────────────────────────────────────
VALID_INTENTS = {
    "risk_analysis",        # Risk factors, threats, litigation
    "revenue_summary",      # Revenue, earnings, financial performance
    "business_overview",    # Company description, segments, products
    "general",              # Catch-all for anything else
}

# ── Ticker + year config — UPDATE THESE when new filings are ingested ─────────
# To add a new ticker: add entry here + run pipeline.py --ticker X --max_filings N
# To add a new year:   add to SUPPORTED_YEARS + run pipeline.py --ticker X --fiscal_year Y
SUPPORTED_TICKERS: dict = {
    "Apple"    : "AAPL",
    "Microsoft": "MSFT",
    "Google"   : "GOOGL",
    "Alphabet" : "GOOGL",
    "Amazon"   : "AMZN",
    "Nvidia"   : "NVDA",
    "Meta"     : "META",
}

SUPPORTED_YEARS: list = [2025]   # extend as new fiscal years are ingested


def _build_system_prompt() -> str:
    """Build system prompt dynamically — reflects currently ingested tickers/years."""
    ticker_map  = ", ".join(f"{k} → {v}" for k, v in SUPPORTED_TICKERS.items())
    valid_ticks = ", ".join(sorted(set(SUPPORTED_TICKERS.values())))
    years_list  = ", ".join(str(y) for y in sorted(SUPPORTED_YEARS))
    return f"""You are a financial query analyzer. Extract structured information from user queries about SEC filings.

Return ONLY a valid JSON object with these fields:
{{
  "ticker": "<stock ticker symbol in uppercase, or null if not found>",
  "fiscal_year": <4-digit fiscal year as integer, or null if not found>,
  "intent": "<one of: risk_analysis | revenue_summary | business_overview | general>"
}}

Rules:
- Convert company names to tickers: {ticker_map}
- Only extract tickers from this supported list: {valid_ticks}
- fiscal_year is the year the 10-K COVERS, not the filing date
- Only extract fiscal years from this supported list: {years_list}
- Convert relative years: "last year" → current year minus 1
- If no fiscal year is mentioned, set fiscal_year to null. Do NOT guess or default to any year.
- If ticker or fiscal_year is not in the supported lists, set to null
- Default intent to "general" if unclear
- Return ONLY the JSON object, no explanation
"""

SYSTEM_PROMPT = _build_system_prompt()


def query_analyzer_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — reads query from state, writes ticker/fiscal_year/intent.

    Args:
        state: Current AgentState passed by LangGraph runtime

    Returns:
        Partial AgentState update with ticker, fiscal_year, intent populated
    """
    query = state["query"]
    logger.info(f"QueryAnalyzer received query: {query}")

    try:
        # ── Call LLM for structured extraction ────────────────────────────────
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ])

        # ── Parse JSON response ───────────────────────────────────────────────
        raw = response.content.strip()

        # Strip markdown code fences if LLM wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        extracted = json.loads(raw)

        # ── Validate and sanitize extracted fields ────────────────────────────
        ticker      = extracted.get("ticker")
        fiscal_year = extracted.get("fiscal_year") or state.get("fiscal_year")  # ← ADD fallback
        intent      = extracted.get("intent", "general")

        # Ensure intent is one of the supported values
        if intent not in VALID_INTENTS:
            intent = "general"

        logger.info(f"QueryAnalyzer extracted → ticker={ticker}, fiscal_year={fiscal_year}, intent={intent}")

        return {
            "ticker":      ticker or state.get("ticker"),
            "fiscal_year": fiscal_year,
            "intent":      intent,
        }

    except (json.JSONDecodeError, KeyError) as e:
        # ── Graceful fallback — never crash the pipeline ──────────────────────
        logger.warning(f"QueryAnalyzer parse failed: {e}. Using defaults.")
        return {
            "ticker":      None,
            "fiscal_year": None, # API layer will reject and surface error
            "intent":      "general",
        }
