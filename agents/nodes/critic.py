"""
agents/nodes/critic.py
-----------------------
Critic Node — fourth and final node in the LangGraph pipeline.

Responsibility:
    Evaluates the draft answer produced by MarketAnalyst and decides:
    - PASS  : quality_score >= 0.7 → set final_answer, route to END
    - RETRY : quality_score <  0.7 → increment retry_count, route back to SECRetriever

Why a Critic node?
    Without a quality gate, bad answers (hallucinations, vague responses,
    insufficient context) reach the user silently. The Critic enforces
    a minimum quality bar and triggers self-correction automatically.

Max retries = 2 to prevent infinite loops. After 2 retries, the best
available draft is accepted regardless of score.

Output:
    Updates AgentState with: quality_score, final_answer
"""

import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AgentState

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── LLM client — gpt-4o-mini sufficient for scoring tasks ────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,       # Fully deterministic scoring
    max_tokens=150,      # Only needs a short JSON score response
)

# ── Quality threshold and retry limit ─────────────────────────────────────────
QUALITY_THRESHOLD = 0.7   # Answers scoring below this trigger a retry
MAX_RETRIES       = 2     # Safety valve — prevents infinite retry loops

# ── Scoring prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a financial answer quality evaluator.

Score the answer on these 4 dimensions (each 0.0-1.0):
1. relevance    : Does it directly answer the question asked?
2. specificity  : Does it cite specific facts, numbers, or dates from the filing?
3. completeness : Does it cover the main aspects of the question?
4. groundedness : Is it based on the provided context (not hallucinated)?

Return ONLY a valid JSON object:
{
  "relevance": <float>,
  "specificity": <float>,
  "completeness": <float>,
  "groundedness": <float>,
  "overall": <float>,
  "reasoning": "<one sentence explaining the score>"
}

The overall score should reflect the weighted average, with groundedness weighted highest.
"""


def critic_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — scores draft answer and sets final_answer if quality passes.

    Routing logic (defined in graph.py via conditional edge):
        quality_score >= 0.7 OR retry_count >= MAX_RETRIES → END
        quality_score <  0.7 AND retry_count < MAX_RETRIES → SECRetriever (retry)

    Args:
        state: Current AgentState with draft_answer populated

    Returns:
        Partial AgentState update with quality_score and final_answer
    """
    query        = state["query"]
    draft_answer = state.get("draft_answer") or ""
    retry_count  = state.get("retry_count")  or 0

    logger.info(f"Critic evaluating answer (retry_count={retry_count})")

    try:
        # ── Ask LLM to score the draft answer ─────────────────────────────────
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {query}\n\nAnswer to evaluate:\n{draft_answer}"),
        ])

        # ── Parse scoring response ────────────────────────────────────────────
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        scores        = json.loads(raw)
        quality_score = float(scores.get("overall", 0.5))
        reasoning     = scores.get("reasoning", "")

        logger.info(f"Critic scored: {quality_score:.2f} — {reasoning}")

        # ── Quality gate decision ─────────────────────────────────────────────
        if quality_score >= QUALITY_THRESHOLD or retry_count >= MAX_RETRIES:
            # PASS — accept draft as final answer
            if retry_count >= MAX_RETRIES:
                logger.warning(f"Max retries reached ({MAX_RETRIES}). Accepting best draft.")
            return {
                "quality_score": quality_score,
                "final_answer":  draft_answer,   # Promote draft to final
            }
        else:
            # RETRY — signal graph to loop back to SECRetriever
            logger.info(f"Quality below threshold ({quality_score:.2f} < {QUALITY_THRESHOLD}). Retrying.")
            return {
                "quality_score": quality_score,
                "retry_count":   retry_count + 1,
                "final_answer":  None,            # Keep None to signal retry needed
            }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # ── Fallback — accept draft on scoring failure ────────────────────────
        logger.warning(f"Critic scoring failed: {e}. Accepting draft as final.")
        return {
            "quality_score": 0.5,
            "final_answer":  draft_answer,
        }
