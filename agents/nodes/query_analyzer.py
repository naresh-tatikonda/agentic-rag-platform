"""
agents/nodes/query_analyzer.py
-------------------------------
QueryAnalyzer Node — the first node in the LangGraph pipeline.

Responsibility:
    Takes the raw user query and, in a single structured-output LLM call,
    extracts tickers/fiscal_year/intent AND decides where the query should
    route:
      - "sec"     : answerable from ingested 10-K filings -> SECRetriever
      - "abstain" : no supported ticker found, or the intent needs data this
                    system doesn't retrieve (e.g. live price performance) ->
                    Abstain node, skips retrieval/generation entirely

    Why decide route here instead of downstream?
        The extraction call already knows tickers + intent; deriving route
        from that in the same call avoids a second LLM round trip. The
        decision logic itself (`derive_route`) is a pure function with no
        I/O, so it's unit-testable without mocking an LLM.

    Why structured output (instructor) instead of manual JSON parsing?
        A single malformed field used to fail the whole hand-rolled
        json.loads() parse. instructor validates against the Pydantic
        schema and re-prompts automatically on a bad response.

Output:
    Updates AgentState with: tickers, fiscal_year, intent, route, abstain_reason
"""

import logging
import os
from typing import Literal, Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from agents.state import AgentState

logger = logging.getLogger(__name__)

# -- LLM client (gpt-4o-mini for cost efficiency on extraction tasks) --------
client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# -- Supported intents ---------------------------------------------------------
# Intents this system can actually answer from ingested 10-K filings.
# "price_performance" is deliberately NOT here — this system has no market
# data source yet, so those queries route to abstain rather than being
# answered (badly) from filing text.
SEC_ANSWERABLE_INTENTS = {
    "risk_analysis",
    "revenue_summary",
    "business_overview",
    "comparison",
    "general",
}

# -- Ticker + year config — UPDATE THESE when new filings are ingested -------
SUPPORTED_TICKERS: dict = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Nvidia": "NVDA",
    "Meta": "META",
}
SUPPORTED_TICKER_VALUES = set(SUPPORTED_TICKERS.values())
SUPPORTED_YEARS: list = [2025]   # extend as new fiscal years are ingested
DEFAULT_FISCAL_YEAR = max(SUPPORTED_YEARS)


class QueryAnalysis(BaseModel):
    """Structured extraction target for the analyzer LLM call."""

    tickers: list[str] = Field(
        default_factory=list,
        description="Stock tickers mentioned or implied by company name, uppercase, e.g. ['AAPL']",
    )
    fiscal_year: Optional[int] = Field(
        default=None,
        description="4-digit fiscal year the question refers to, or null if not mentioned",
    )
    intent: Literal[
        "risk_analysis",
        "revenue_summary",
        "business_overview",
        "price_performance",
        "comparison",
        "general",
    ] = Field(description="Primary intent of the query")


def _build_system_prompt() -> str:
    """Build system prompt dynamically — reflects currently ingested tickers/years."""
    ticker_map = ", ".join(f"{k} -> {v}" for k, v in SUPPORTED_TICKERS.items())
    years_list = ", ".join(str(y) for y in sorted(SUPPORTED_YEARS))
    return f"""You are a financial query analyzer for a system that answers questions from SEC 10-K filings.

Extract tickers, fiscal_year, and intent from the user's query.

Rules:
- Convert company names to tickers: {ticker_map}
- A query can mention more than one company (e.g. "compare AMD and AVGO") -> return all tickers found
- fiscal_year is the year the 10-K COVERS, not the filing date. Convert relative years
  ("last year" -> current year minus 1). Only fill fiscal_year if the query states or clearly implies one;
  leave it null otherwise (the caller defaults to the most recently ingested year: {years_list}).
- intent = "price_performance" for anything about stock price, returns, or how a stock "did" over a period —
  do NOT force these into revenue_summary just because a ticker is present.
- intent = "comparison" whenever two or more companies are being compared, regardless of what's being compared.
- Default intent to "general" if unclear.
"""


SYSTEM_PROMPT = _build_system_prompt()


def derive_route(
    tickers: list[str],
    intent: str,
    fiscal_year: Optional[int],
) -> tuple[str, Optional[str], list[str], int]:
    """
    Pure decision function — no I/O, no LLM call. Unit-testable in isolation.

    Returns:
        (route, abstain_reason, valid_tickers, resolved_fiscal_year)
    """
    valid_tickers = [t for t in tickers if t in SUPPORTED_TICKER_VALUES]
    resolved_fiscal_year = fiscal_year if fiscal_year is not None else DEFAULT_FISCAL_YEAR

    if not valid_tickers:
        return "abstain", "no supported ticker identified in the query", valid_tickers, resolved_fiscal_year

    if intent not in SEC_ANSWERABLE_INTENTS:
        reason = (
            f"intent '{intent}' requires live market data, which this system doesn't retrieve — "
            "it only answers from ingested SEC 10-K filings"
        )
        return "abstain", reason, valid_tickers, resolved_fiscal_year

    if resolved_fiscal_year not in SUPPORTED_YEARS:
        reason = f"no ingested 10-K for fiscal year {resolved_fiscal_year}"
        return "abstain", reason, valid_tickers, resolved_fiscal_year

    return "sec", None, valid_tickers, resolved_fiscal_year


def query_analyzer_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — reads query from state, writes
    tickers/fiscal_year/intent/route/abstain_reason.
    """
    query = state["query"]
    logger.info(f"QueryAnalyzer received query: {query}")

    try:
        analysis = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_retries=2,
            response_model=QueryAnalysis,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
        )

        # A caller-supplied hint (e.g. the RAGAS eval runner passing a known
        # ticker/fiscal_year) fills in only what the LLM didn't extract from
        # the query text itself — it never overrides an actual extraction.
        tickers = analysis.tickers or (state.get("tickers") or [])
        fiscal_year_hint = analysis.fiscal_year or state.get("fiscal_year")

        route, abstain_reason, valid_tickers, fiscal_year = derive_route(
            tickers, analysis.intent, fiscal_year_hint
        )

        logger.info(
            f"QueryAnalyzer extracted -> tickers={valid_tickers}, fiscal_year={fiscal_year}, "
            f"intent={analysis.intent}, route={route}"
        )

        return {
            "tickers": valid_tickers,
            "fiscal_year": fiscal_year,
            "intent": analysis.intent,
            "route": route,
            "abstain_reason": abstain_reason,
        }

    except Exception as e:
        # Fail closed: a broken extraction call must not fall through to
        # retrieval/generation with garbage state — abstain instead.
        logger.warning(f"QueryAnalyzer failed: {e}. Routing to abstain.")
        return {
            "tickers": [],
            "fiscal_year": None,
            "intent": "general",
            "route": "abstain",
            "abstain_reason": "query analysis failed",
        }
