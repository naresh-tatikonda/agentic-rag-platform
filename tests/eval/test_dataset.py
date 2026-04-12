# tests/test_dataset.py
#
# PURPOSE: Golden evaluation dataset for the agentic-rag-platform RAGAS eval suite.
#
# DATA SOURCE: Apple Inc. Form 10-K FY2023 (EDGAR) — the ONLY ticker/year
#              currently ingested in the pgvector store.
#
# WHY ONLY 3 CASES:
#   Quality over quantity for a portfolio project. Each case targets a
#   different RAGAS failure mode:
#     tc_001 → Faithfulness (will LLM hallucinate a number?)
#     tc_002 → Context Recall (will retriever find ALL risk sections?)
#     tc_003 → Multi-chunk synthesis (hardest — stresses all 4 metrics)
#
# ADDING MORE CASES LATER:
#   When more tickers/years are ingested, append to EVAL_DATASET.
#   Do NOT modify existing cases — regressions are detected by case_id.
#
# RAGAS INPUT CONTRACT (all 4 fields required at eval time):
#   question     : raw user query
#   answer       : pipeline response (populated by eval runner, NOT here)
#   contexts     : retrieved chunks (populated by eval runner, NOT here)
#   ground_truth : human-written ideal answer (fixed here — never changes)

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalCase:
    """
    One RAGAS test case.

    Fields populated HERE (static):
        case_id, difficulty, question, ground_truth, contexts

    Fields populated at EVAL TIME by Module 2 (eval runner):
        answer  ← comes from live pipeline response
    """
    case_id      : str
    difficulty   : str        # 'easy' | 'medium' | 'hard'
    question     : str
    ground_truth : str
    contexts     : List[str]  # what a GOOD retriever should return
    answer       : str = ""   # empty here; filled by eval runner


# ── Golden Dataset — AAPL FY2023 Only ────────────────────────────────────────

EVAL_DATASET: List[EvalCase] = [

    EvalCase(
        case_id    = "tc_001",
        difficulty = "easy",

        # Factual single-number lookup — primary RAGAS target: Faithfulness
        # If LLM returns anything other than $383.3B it is hallucinating
        # Your pipeline already returned this correctly (quality_score=0.95)
        question   = "What was Apple's total net revenue in fiscal year 2023?",

        ground_truth = (
            "Apple's total net revenue for fiscal year 2023 was $383,285 million "
            "($383.3 billion), a decrease from $394,328 million in fiscal year 2022."
        ),

        # Single chunk — the revenue table from the 10-K income statement
        # Context Precision should be 1.0 here (one chunk, fully relevant)
        contexts = [
            (
                "Apple Inc. Form 10-K FY2023 Consolidated Statements of Operations: "
                "Net sales — Products $298,085M, Services $85,200M, "
                "Total net sales $383,285M (2023) vs $394,328M (2022) vs $365,817M (2021)."
            )
        ],
    ),

    EvalCase(
        case_id    = "tc_002",
        difficulty = "medium",

        # Multi-chunk, single-doc — primary RAGAS target: Context Recall
        # Retriever must find iPhone + Mac + iPad + Wearables + Services chunks
        # Your pipeline returned all 5 product lines correctly — good signal
        question   = "What were Apple's revenue figures broken down by product category in FY2023?",

        ground_truth = (
            "Apple's FY2023 revenue by product category: "
            "iPhone $200,583M, Mac $29,357M, iPad $28,300M, "
            "Wearables/Home/Accessories $39,845M, Services $85,200M, "
            "Total $383,285M."
        ),

        # 3 chunks representing different sections of the 10-K product breakdown
        # Tests whether retriever returns ALL relevant chunks (recall)
        # AND whether it avoids irrelevant chunks (precision)
        contexts = [
            (
                "Apple 10-K FY2023 Product Net Sales: iPhone $200,583M (2023) "
                "vs $205,489M (2022). iPhone represents the largest revenue segment."
            ),
            (
                "Apple 10-K FY2023 Product Net Sales: Mac $29,357M, iPad $28,300M, "
                "Wearables Home and Accessories $39,845M for fiscal year 2023."
            ),
            (
                "Apple 10-K FY2023 Segment Data: Services net sales $85,200M in 2023 "
                "vs $78,129M in 2022. Services include advertising, AppleCare, "
                "cloud services, digital content, and payment services."
            ),
        ],
    ),

    EvalCase(
        case_id    = "tc_003",
        difficulty = "hard",

        # Multi-chunk synthesis — stresses ALL 4 RAGAS metrics simultaneously
        # LLM must connect gross margin data WITH strategic narrative
        # This is the case most likely to expose hallucination or drift
        question   = (
            "What drove Apple's gross margin improvement from FY2021 to FY2023, "
            "and how does this relate to Apple's Services strategy?"
        ),

        ground_truth = (
            "Apple's gross margin improved from 41.8% in FY2021 to 44.1% in FY2023. "
            "This was primarily driven by the growth of the Services segment, which "
            "carried a gross margin of approximately 70.8% in FY2023 compared to "
            "Products gross margin of 36.6%. As Services grew as a proportion of "
            "total revenue, it structurally lifted the blended gross margin. "
            "Apple's 10-K explicitly identifies Services growth as a strategic priority."
        ),

        # Two chunk types from different sections of the same 10-K:
        #   1. Financial data (margin table)
        #   2. Strategic narrative (MD&A section)
        # Retriever must find BOTH — tests cross-section recall
        contexts = [
            (
                "Apple 10-K FY2023 Gross Margin: Total gross margin 44.1% (2023), "
                "43.3% (2022), 41.8% (2021). Products gross margin 36.6% (2023). "
                "Services gross margin 70.8% (2023)."
            ),
            (
                "Apple 10-K FY2023 MD&A — Services: Services revenue grew from "
                "$78,129M in FY2022 to $85,200M in FY2023. Management views Services "
                "as the highest-margin and fastest-growing segment, central to "
                "long-term profitability strategy."
            ),
        ],
    ),
]


# ── Sanity check ──────────────────────────────────────────────────────────────
# Run directly to verify dataset loads without import errors:
#   python tests/test_dataset.py
if __name__ == "__main__":
    print(f"✅ Loaded {len(EVAL_DATASET)} eval cases\n")
    for case in EVAL_DATASET:
        print(
            f"  [{case.difficulty.upper():6}] {case.case_id} — "
            f"{case.question[:65]}..."
        )
    print("\n⚠️  NOTE: 'answer' fields are empty — populated at eval time by Module 2")
