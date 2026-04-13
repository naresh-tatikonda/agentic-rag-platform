# tests/eval/test_dataset.py
#
# PURPOSE: Golden evaluation dataset for the agentic-rag-platform RAGAS eval suite.
#
# DATA SOURCE: Apple Inc. Form 10-K FY2025 (EDGAR) — the ONLY ticker/year
#              currently ingested in the pgvector store.
#
# WHY ONLY 3 CASES:
#   Quality over quantity for a portfolio project. Each case targets a
#   different RAGAS failure mode:
#     tc_001 → Faithfulness (will LLM hallucinate a number?)
#     tc_002 → Context Recall (will retriever find ALL revenue sections?)
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
#
# SOURCE NUMBERS — verified from FY2025 10-K retrieved chunks (curl confirmed):
#   Total net sales:        $416,161M
#   iPhone:                 $209,586M
#   Mac:                    $33,708M
#   iPad:                   $28,023M
#   Wearables/Home/Acc:     $35,686M
#   Services:               $109,158M
#   Total gross margin:     $195,201M / 46.9%
#   Products gross margin:  36.8%
#   Services gross margin:  75.4%
#   R&D:                    $34,550M

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalCase:
    """
    One RAGAS test case.

    Fields populated HERE (static):
        case_id, difficulty, question, ground_truth, contexts

    Fields populated at EVAL TIME by eval runner:
        answer  ← comes from live pipeline response
    """
    case_id      : str
    difficulty   : str        # 'easy' | 'medium' | 'hard'
    question     : str
    ground_truth : str
    contexts     : List[str]
    ticker       : str = "AAPL"
    fiscal_year  : int = 2025
    answer       : str = ""


# ── Golden Dataset — AAPL FY2025 Only ────────────────────────────────────────

EVAL_DATASET: List[EvalCase] = [

    EvalCase(
        case_id    = "tc_001",
        difficulty = "easy",

        # Factual single-number lookup — primary RAGAS target: Faithfulness
        # If LLM returns anything other than $416.2B it is hallucinating
        question   = "What was Apple's total net revenue in fiscal year 2025?",

        ground_truth = (
            "Apple's total net revenue for fiscal year 2025 was $416,161 million "
            "($416.2 billion), an increase of 6% from $391,035 million in fiscal year 2024."
        ),

        # Single chunk — revenue table from FY2025 10-K income statement
        contexts = [
            (
                "Apple Inc. Form 10-K FY2025 Products and Services Performance: "
                "iPhone $209,586M, Mac $33,708M, iPad $28,023M, "
                "Wearables/Home/Accessories $35,686M, Services $109,158M, "
                "Total net sales $416,161M (2025) vs $391,035M (2024) vs $383,285M (2023)."
            )
        ],
    ),

    EvalCase(
        case_id    = "tc_002",
        difficulty = "medium",

        # Multi-chunk — primary RAGAS target: Context Recall
        # Retriever must find all 5 product line chunks
        question   = "What were Apple's revenue figures broken down by product category in FY2025?",

        ground_truth = (
            "Apple's FY2025 revenue by product category: "
            "iPhone $209,586M (+4% YoY), Mac $33,708M (+12% YoY), "
            "iPad $28,023M (+5% YoY), Wearables/Home/Accessories $35,686M (-4% YoY), "
            "Services $109,158M (+14% YoY), Total $416,161M (+6% YoY)."
        ),

        contexts = [
            (
                "Apple 10-K FY2025 iPhone net sales: $209,586M (2025) vs $201,183M (2024). "
                "iPhone net sales increased due to higher net sales of Pro models."
            ),
            (
                "Apple 10-K FY2025 Product Net Sales: Mac $33,708M (+12%), "
                "iPad $28,023M (+5%), Wearables Home and Accessories $35,686M (-4%) "
                "for fiscal year 2025."
            ),
            (
                "Apple 10-K FY2025 Services net sales $109,158M (+14%) vs $96,169M (2024). "
                "Growth driven by higher net sales from advertising, App Store, "
                "and cloud services."
            ),
        ],
    ),

    EvalCase(
        case_id    = "tc_003",
        difficulty = "hard",

        # Multi-chunk synthesis — primary RAGAS target: all 4 metrics
        # Tests whether LLM connects gross margin table with Services narrative
        question = (
            "What drove Apple's gross margin improvement in FY2025 "
            "and how does this relate to Apple's Services strategy?"
        ),

        ground_truth = (
            "Apple's total gross margin was 46.9% in FY2025, up from 46.2% in FY2024. "
            "This improvement was driven by Services segment growth — Services gross "
            "margin reached 75.4% in FY2025 vs Products at 36.8%, structurally lifting "
            "the blended rate. Services net sales grew 14% to $109,158M, driven by "
            "advertising, App Store, and cloud services. The high-margin Services mix "
            "offset tariff cost pressure on Products gross margin percentage, which "
            "declined slightly from 37.2% to 36.8% due to tariff costs and product mix."
        ),

        contexts = [
            (
                "Apple 10-K FY2025 Gross Margin Table: "
                "Products gross margin $112,887M / 36.8% (2025) vs 37.2% (2024). "
                "Services gross margin $82,314M / 75.4% (2025) vs 73.9% (2024). "
                "Total gross margin $195,201M / 46.9% (2025) vs 46.2% (2024)."
            ),
            (
                "Apple 10-K FY2025 Services Gross Margin: increased during 2025 compared "
                "to 2024 primarily due to higher Services net sales and a different mix "
                "of services. Services gross margin percentage increased due to a different "
                "mix of services, partially offset by higher costs."
            ),
            (
                "Apple 10-K FY2025 Products Gross Margin: increased during 2025 compared "
                "to 2024 primarily due to favorable costs and a different mix of products, "
                "partially offset by tariff costs. Products gross margin percentage decreased "
                "due to a different mix of products and tariff costs."
            ),
        ],
    ),

]
