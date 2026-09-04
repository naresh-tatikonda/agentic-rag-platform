# tests/agents/test_critic.py
#
# Tests the critic's pure decision functions — grounding check, quality
# score, and pass/retry/abstain outcome — plus a mocked test of the full
# node to verify it fails CLOSED when the LLM scoring call errors (never
# promotes an unverified draft to final_answer).

from unittest.mock import patch

from agents.nodes.critic import (
    is_claim_grounded,
    grade_claims,
    compute_quality_score,
    decide_outcome,
    critic_node,
    QUALITY_THRESHOLD,
    MAX_RETRIES,
)


# -- is_claim_grounded --------------------------------------------------------

def test_claim_with_matching_number_is_grounded():
    assert is_claim_grounded("Revenue was $416,161M in FY2025.", "Total net sales were $416,161 million in fiscal 2025.")


def test_claim_with_fabricated_number_is_not_grounded():
    assert not is_claim_grounded("Revenue was $999,999M.", "Total net sales were $416,161 million.")


def test_claim_with_no_numbers_is_grounded_by_default():
    # Known limitation, documented in critic.py — no numeric content to check.
    assert is_claim_grounded("Apple describes itself as a consumer electronics company.", "Apple designs and sells consumer electronics.")


def test_claim_with_no_source_text_is_not_grounded():
    assert not is_claim_grounded("Revenue was $416,161M.", None)


# -- grade_claims --------------------------------------------------------------

def test_grade_claims_splits_grounded_and_ungrounded():
    chunks_by_id = {"AAPL:1": {"text": "Revenue was $416,161M."}}
    claims = [
        {"text": "Revenue was $416,161M.", "source_chunk_id": "AAPL:1"},
        {"text": "Revenue was $999M.", "source_chunk_id": "AAPL:1"},
    ]

    fraction, grounded, ungrounded = grade_claims(claims, chunks_by_id)

    assert fraction == 0.5
    assert len(grounded) == 1
    assert len(ungrounded) == 1


def test_grade_claims_empty_list_is_zero_not_error():
    assert grade_claims([], {}) == (0.0, [], [])


def test_claim_citing_nonexistent_chunk_is_ungrounded():
    fraction, grounded, ungrounded = grade_claims(
        [{"text": "Revenue was $1M.", "source_chunk_id": "does-not-exist"}], {}
    )
    assert fraction == 0.0
    assert len(ungrounded) == 1


# -- compute_quality_score / decide_outcome -------------------------------------

def test_quality_score_weights_groundedness_highest():
    grounded_heavy = compute_quality_score(grounded_fraction=1.0, relevance=0.0, completeness=0.0)
    relevance_heavy = compute_quality_score(grounded_fraction=0.0, relevance=1.0, completeness=1.0)
    assert grounded_heavy > relevance_heavy


def test_perfect_score_passes():
    assert decide_outcome(1.0, retry_count=0) == "pass"


def test_low_score_with_retries_left_retries():
    assert decide_outcome(0.1, retry_count=0) == "retry"


def test_low_score_with_retries_exhausted_abstains():
    assert decide_outcome(0.1, retry_count=MAX_RETRIES) == "abstain"


def test_threshold_boundary_is_inclusive():
    assert decide_outcome(QUALITY_THRESHOLD, retry_count=0) == "pass"


# -- critic_node fails closed on a broken scoring call --------------------------

def test_critic_node_fails_closed_when_llm_scoring_call_errors():
    state = {
        "query": "What was AAPL revenue?",
        "draft_claims": [{"text": "Revenue was $416,161M.", "source_chunk_id": "AAPL:1"}],
        "retrieved_chunks": [{"chunk_id": "AAPL:1", "ticker": "AAPL", "text": "Total net sales were $416,161M."}],
        "retry_count": 0,
    }

    with patch("agents.nodes.critic.client") as mock_client:
        mock_client.chat.completions.create.side_effect = TimeoutError("scoring call timed out")
        result = critic_node(state)

    # groundedness alone (1.0) * GROUNDEDNESS_WEIGHT (0.6) = 0.6, below the 0.7
    # threshold — a broken scorer must not be able to inflate this to a pass.
    assert result["quality_score"] < QUALITY_THRESHOLD
    assert result["final_answer"] is None   # retry, not a silently-promoted draft


def test_critic_node_abstains_with_honest_message_when_retries_exhausted():
    state = {
        "query": "What was AAPL revenue?",
        "draft_claims": [{"text": "Revenue was $999M.", "source_chunk_id": "AAPL:1"}],  # fabricated
        "retrieved_chunks": [{"chunk_id": "AAPL:1", "ticker": "AAPL", "text": "Total net sales were $416,161M."}],
        "retry_count": MAX_RETRIES,
        "tickers": ["AAPL"],
        "fiscal_year": 2025,
    }

    with patch("agents.nodes.critic.client") as mock_client:
        mock_client.chat.completions.create.side_effect = TimeoutError("scoring call timed out")
        result = critic_node(state)

    assert result["route"] == "abstain"
    assert result["final_answer"] is not None
    assert "AAPL" in result["final_answer"]
