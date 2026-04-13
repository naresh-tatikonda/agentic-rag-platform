# scripts/ci_ragas_eval.py
# =============================================================================
# RAGAS CI Quality Gate
# Runs on GitHub Actions after pytest passes, before docker build.
#
# Purpose:
#   Catches LLM quality regressions before they reach production.
#   A code change that drops faithfulness below threshold blocks deployment.
#
# How it works:
#   1. Load 10 golden questions from scripts/golden_test_set.json
#   2. Call the live FastAPI /query endpoint for each question
#   3. Collect: question, answer, retrieved_chunks (contexts)
#   4. Score with RAGAS faithfulness metric
#   5. Exit 0 (pass) if score >= threshold, Exit 1 (fail) if below
#
# RAGAS faithfulness measures:
#   "Can every claim in the answer be traced back to the retrieved chunks?"
#   Score range: 0.0 (hallucination) → 1.0 (fully grounded)
# =============================================================================

import os
import sys
import json
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness


# ── Configuration — all values injected from GitHub Secrets at CI runtime ────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY      = os.environ.get("API_KEY", "")
THRESHOLD    = float(os.environ.get("RAGAS_FAITHFULNESS_THRESHOLD", "0.7"))

# Path to the fixed golden test set — committed to the repo
GOLDEN_SET_PATH = os.path.join(
    os.path.dirname(__file__), "golden_test_set.json"
)


def load_golden_test_set() -> list[dict]:
    """
    Load the fixed golden test questions from JSON.
    These questions are hand-curated to cover critical RAG query types:
    - Single ticker lookup
    - Fiscal year comparison
    - Multi-hop reasoning
    - Edge cases (missing data, ambiguous queries)
    """
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def query_api(question: str, ticker: str, fiscal_year: int) -> dict | None:
    """
    Call the live FastAPI /query endpoint for one golden question.
    Returns the full response dict or None on failure.

    We call the LIVE API (not a mock) because we want to test the full
    pipeline: retriever → LLM → critic. Mocking would defeat the purpose.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": question,
                "ticker": ticker,
                "fiscal_year": fiscal_year,
            },
            headers={"X-API-Key": API_KEY},
            # 60s timeout — LLM calls can be slow under load
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  ✗ API call failed: {e}")
        return None


def build_ragas_dataset(golden_set: list[dict]) -> Dataset | None:
    """
    Run all golden questions through the API and build a RAGAS Dataset.

    RAGAS faithfulness needs three fields per row:
      - question:  the input query
      - answer:    the LLM's final answer
      - contexts:  list of retrieved chunk texts used to generate the answer

    If too many API calls fail, returns None to abort evaluation.
    """
    questions = []
    answers   = []
    contexts  = []
    failures  = 0

    for i, item in enumerate(golden_set):
        print(f"  [{i+1}/{len(golden_set)}] {item['question'][:60]}...")

        result = query_api(
            question=item["question"],
            ticker=item["ticker"],
            fiscal_year=item["fiscal_year"],
        )

        if result is None:
            failures += 1
            continue

        questions.append(item["question"])
        answers.append(result.get("final_answer", ""))
        # retrieved_chunks is a list of chunk text strings
        # populated by SECRetrieverNode and returned in the API response
        contexts.append(result.get("retrieved_chunks", []))

    # Abort if more than 30% of calls failed — results would be unreliable
    failure_rate = failures / len(golden_set)
    if failure_rate > 0.3:
        print(f"\n✗ Too many API failures ({failures}/{len(golden_set)}). Aborting.")
        return None

    return Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    })


def main() -> None:
    """
    Main CI gate logic:
    1. Load golden test set
    2. Build RAGAS dataset via live API calls
    3. Evaluate faithfulness
    4. Exit 0 (pass) or 1 (fail) based on threshold
    """
    print("=" * 60)
    print("RAGAS CI Quality Gate")
    print(f"API:       {API_BASE_URL}")
    print(f"Threshold: faithfulness >= {THRESHOLD}")
    print("=" * 60)

    # Step 1: Load golden questions
    print("\nLoading golden test set...")
    golden_set = load_golden_test_set()
    print(f"  Loaded {len(golden_set)} questions")

    # Step 2: Call API for each question
    print("\nQuerying live API...")
    dataset = build_ragas_dataset(golden_set)

    if dataset is None:
        print("\n✗ RAGAS evaluation aborted — too many API failures")
        sys.exit(1)

    # Step 3: Score with RAGAS faithfulness
    print(f"\nScoring {len(dataset)} responses with RAGAS faithfulness...")
    results = evaluate(dataset, metrics=[faithfulness])

    score = results["faithfulness"]
    print(f"\n{'='*60}")
    print(f"Faithfulness score: {score:.3f}")
    print(f"Threshold:          {THRESHOLD}")

    # Step 4: CI gate decision
    if score >= THRESHOLD:
        print(f"✅ PASSED — score {score:.3f} >= {THRESHOLD}")
        print("=" * 60)
        sys.exit(0)   # Exit 0 = success → pipeline continues to docker build
    else:
        print(f"❌ FAILED — score {score:.3f} < {THRESHOLD}")
        print("Deployment blocked. Fix LLM quality regression before merging.")
        print("=" * 60)
        sys.exit(1)   # Exit 1 = failure → pipeline stops, no deployment


if __name__ == "__main__":
    main()
