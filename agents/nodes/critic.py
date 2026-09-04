"""
agents/nodes/critic.py
-----------------------
Critic Node — fifth and final node in the LangGraph pipeline (route == "sec").

Responsibility:
    Grounds each claim from MarketAnalyst against the chunk it cited, then
    decides:
      - PASS    : quality_score >= threshold -> final_answer built from the
                  GROUNDED claims only (ungrounded ones are dropped, not shipped)
      - RETRY   : quality_score < threshold, retries remain -> loop back to
                  SECRetriever
      - ABSTAIN : quality_score < threshold, retries exhausted -> honest
                  "insufficient evidence" message, never the raw draft

    This is a deliberate behavior change from a prior version of this node,
    which accepted the best available draft once retries ran out regardless
    of score. For a financial-answers system, shipping a low-confidence
    guess is worse than saying "I don't know" — so retries-exhausted now
    routes to abstain, not to a forced pass.

Why a deterministic grounding check instead of asking an LLM "is this
grounded?"
    An LLM checking its own sibling call's output for hallucination is a
    weak self-critique pattern — it can rubber-stamp a fabricated number
    with the same confidence it fabricated it. The grounding check here is
    a plain string/number match: every numeric token in a claim (dollar
    amounts, percentages, years) must literally appear in the chunk it
    cites. Cheap, can't itself hallucinate, and it's exactly the failure
    mode that matters most for financial data — invented numbers.
    Known limitation: a claim with no numeric content that's simply
    off-topic isn't caught by this check alone; that's covered by the
    relevance/completeness LLM score below, not by groundedness. A full
    NLI/entailment check on non-numeric claims is a natural next step,
    not built here to avoid an extra LLM call per claim.

Fail-closed:
    If the relevance/completeness LLM call errors or times out, quality_score
    is computed from groundedness ALONE (down-weighted, never inflated) —
    a broken scorer can only push the outcome toward retry/abstain, never
    toward a pass it didn't earn.

Output:
    Updates AgentState with: quality_score, ungrounded_claims, final_answer,
    and (on abstain) route
"""

import logging
import os
import re

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from agents.state import AgentState

logger = logging.getLogger(__name__)

# 20s timeout bounds the critic's own LLM call — a hung scoring call must
# not hang the whole request; it fails closed into the except branch below.
client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=20.0))

QUALITY_THRESHOLD = 0.7
MAX_RETRIES = 2

GROUNDEDNESS_WEIGHT = 0.6
RELEVANCE_WEIGHT = 0.25
COMPLETENESS_WEIGHT = 0.15

_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*%?")


class QualityAssessment(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0, description="Does the answer address the question asked?")
    completeness: float = Field(ge=0.0, le=1.0, description="Does it cover the main aspects of the question?")
    reasoning: str = Field(description="One sentence explaining the scores")


SYSTEM_PROMPT = """You are a financial answer quality evaluator.
Score ONLY relevance and completeness — groundedness is checked separately, do not consider it.
relevance: does the answer directly address the question asked?
completeness: does it cover the main aspects of the question?
"""


def _extract_numeric_tokens(text: str) -> set[str]:
    return {t.replace(",", "").replace("$", "").replace("%", "").strip() for t in _NUMERIC_TOKEN_RE.findall(text)}


def is_claim_grounded(claim_text: str, source_text: str | None) -> bool:
    """Pure function — every numeric token in the claim must appear in its cited source text."""
    if not source_text:
        return False
    claim_numbers = _extract_numeric_tokens(claim_text)
    if not claim_numbers:
        return True   # no checkable numeric content — see module docstring limitation
    return claim_numbers.issubset(_extract_numeric_tokens(source_text))


def grade_claims(claims: list[dict], chunks_by_id: dict[str, dict]) -> tuple[float, list[dict], list[dict]]:
    """
    Pure function — no I/O. Returns (grounded_fraction, grounded_claims, ungrounded_claims).
    """
    if not claims:
        return 0.0, [], []

    grounded, ungrounded = [], []
    for claim in claims:
        chunk = chunks_by_id.get(claim["source_chunk_id"])
        source_text = chunk["text"] if chunk else None
        if is_claim_grounded(claim["text"], source_text):
            grounded.append(claim)
        else:
            ungrounded.append({**claim, "grounded": False})

    return len(grounded) / len(claims), grounded, ungrounded


def compute_quality_score(grounded_fraction: float, relevance: float, completeness: float) -> float:
    """Pure function. Groundedness weighted highest — see module docstring."""
    return GROUNDEDNESS_WEIGHT * grounded_fraction + RELEVANCE_WEIGHT * relevance + COMPLETENESS_WEIGHT * completeness


def decide_outcome(quality_score: float, retry_count: int) -> str:
    """Pure function. Returns 'pass' | 'retry' | 'abstain'."""
    if quality_score >= QUALITY_THRESHOLD:
        return "pass"
    if retry_count < MAX_RETRIES:
        return "retry"
    return "abstain"


def _build_abstain_message(state: AgentState) -> str:
    tickers = state.get("tickers") or []
    fiscal_year = state.get("fiscal_year")
    return (
        f"I don't have enough grounded evidence in the FY{fiscal_year} SEC filing(s) for "
        f"{', '.join(tickers) or 'the requested company'} to answer this confidently."
    )


def critic_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — grades claims, decides pass/retry/abstain.
    """
    query = state["query"]
    claims = state.get("draft_claims") or []
    chunks_by_id = {c["chunk_id"]: c for c in state.get("retrieved_chunks") or []}
    retry_count = state.get("retry_count") or 0

    grounded_fraction, grounded_claims, ungrounded_claims = grade_claims(claims, chunks_by_id)

    try:
        assessment = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_retries=1,
            response_model=QualityAssessment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nAnswer to evaluate:\n{state.get('draft_answer') or ''}"},
            ],
        )
        quality_score = compute_quality_score(grounded_fraction, assessment.relevance, assessment.completeness)
    except Exception as e:
        logger.warning(f"Critic quality-assessment call failed: {e}. Scoring on groundedness alone (fail closed).")
        quality_score = GROUNDEDNESS_WEIGHT * grounded_fraction

    outcome = decide_outcome(quality_score, retry_count)
    logger.info(
        f"Critic: quality_score={quality_score:.2f} (grounded={grounded_fraction:.2f}), "
        f"outcome={outcome}, retry_count={retry_count}"
    )

    if outcome == "pass":
        final_answer = " ".join(c["text"] for c in grounded_claims)
        return {"quality_score": quality_score, "ungrounded_claims": ungrounded_claims, "final_answer": final_answer}

    if outcome == "retry":
        return {
            "quality_score": quality_score,
            "ungrounded_claims": ungrounded_claims,
            "retry_count": retry_count + 1,
            "final_answer": None,
        }

    # abstain — retries exhausted, still below threshold
    return {
        "quality_score": quality_score,
        "ungrounded_claims": ungrounded_claims,
        "final_answer": _build_abstain_message(state),
        "route": "abstain",
    }
