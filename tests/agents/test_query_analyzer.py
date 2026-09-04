# tests/agents/test_query_analyzer.py
#
# Tests `derive_route` — the pure routing decision function. No LLM call,
# no mocking needed: this is exactly why the decision logic was pulled out
# of query_analyzer_node into its own function.

from agents.nodes.query_analyzer import derive_route, DEFAULT_FISCAL_YEAR


def test_no_ticker_abstains():
    route, reason, tickers, year = derive_route([], "general", None)
    assert route == "abstain"
    assert "ticker" in reason
    assert tickers == []


def test_unsupported_ticker_abstains():
    route, reason, tickers, year = derive_route(["TSLA"], "price_performance", None)
    assert route == "abstain"
    assert tickers == []   # TSLA is not in SUPPORTED_TICKER_VALUES


def test_price_performance_intent_abstains_even_with_valid_ticker():
    route, reason, tickers, year = derive_route(["AAPL"], "price_performance", None)
    assert route == "abstain"
    assert "market data" in reason
    assert tickers == ["AAPL"]   # ticker extraction still succeeded — just out of scope


def test_valid_ticker_and_sec_intent_routes_to_sec():
    route, reason, tickers, year = derive_route(["AAPL"], "risk_analysis", None)
    assert route == "sec"
    assert reason is None
    assert tickers == ["AAPL"]


def test_missing_fiscal_year_defaults_to_most_recent_ingested():
    _, _, _, year = derive_route(["AAPL"], "general", None)
    assert year == DEFAULT_FISCAL_YEAR


def test_uningested_fiscal_year_abstains():
    route, reason, tickers, year = derive_route(["AAPL"], "general", 2019)
    assert route == "abstain"
    assert "2019" in reason
    assert year == 2019


def test_comparison_intent_keeps_multiple_valid_tickers():
    route, reason, tickers, year = derive_route(["AMD", "AAPL", "AVGO"], "comparison", None)
    # AMD/AVGO aren't ingested — only AAPL is a supported ticker today
    assert tickers == ["AAPL"]
    assert route == "sec"


def test_all_tickers_unsupported_abstains_even_for_comparison():
    route, reason, tickers, year = derive_route(["AMD", "AVGO"], "comparison", None)
    assert route == "abstain"
    assert tickers == []
