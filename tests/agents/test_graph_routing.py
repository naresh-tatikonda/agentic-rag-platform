# tests/agents/test_graph_routing.py
#
# Tests the graph's conditional-edge functions directly, with hand-built
# state dicts. These are pure functions over AgentState — no LangGraph
# runtime, no LLM, no DB needed to verify the routing logic.

from agents.graph import route_after_analysis, should_retry
from agents.nodes.abstain import abstain_node


def test_route_after_analysis_sec():
    assert route_after_analysis({"route": "sec"}) == "sec"


def test_route_after_analysis_abstain():
    assert route_after_analysis({"route": "abstain"}) == "abstain"


def test_route_after_analysis_missing_route_defaults_to_abstain():
    # Fail closed: if query_analyzer_node ever returns without setting
    # `route`, the graph must not silently fall through to retrieval.
    assert route_after_analysis({}) == "abstain"


def test_should_retry_ends_when_final_answer_set():
    assert should_retry({"final_answer": "some answer", "quality_score": 0.9}) == "end"


def test_should_retry_ends_on_abstain_message_too():
    # critic_node sets final_answer on ITS abstain path too — should_retry
    # doesn't need to know pass vs. abstain, only whether a final_answer exists.
    assert should_retry({"final_answer": "I can't answer that...", "quality_score": 0.2}) == "end"


def test_should_retry_loops_when_final_answer_is_none():
    assert should_retry({"final_answer": None, "retry_count": 1}) == "retry"


def test_abstain_node_never_calls_out_to_anything():
    result = abstain_node({"abstain_reason": "no supported ticker identified in the query"})
    assert result["quality_score"] == 0.0
    assert "no supported ticker" in result["final_answer"]


def test_abstain_node_has_a_safe_default_reason():
    result = abstain_node({})
    assert result["final_answer"]   # non-empty, doesn't crash on missing reason
