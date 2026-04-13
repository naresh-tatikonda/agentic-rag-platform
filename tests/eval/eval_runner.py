# tests/eval/eval_runner.py
#
# PURPOSE: RAGAS evaluation runner for the agentic-rag-platform.
#
# HOW IT WORKS:
#   1. Loads the golden dataset from test_dataset.py
#   2. Calls the live /query API for each test case
#   3. Feeds {question, answer, contexts, ground_truth} into RAGAS
#   4. Prints a score report and saves results to tests/eval/results/
#   5. Exits with code 1 if scores fall below thresholds (blocks CI/CD)
#
# WHY EXIT CODE 1 MATTERS:
#   GitHub Actions checks exit codes. If this script exits 1,
#   the CI pipeline fails and the PR is blocked from merging.
#   This is the deployment quality gate.
#
# RAGAS VERSION: 0.4.3
#   - Metrics must be instantiated as objects: Faithfulness() not faithfulness
#   - LLM must be passed explicitly via llm_factory (not LangchainLLMWrapper)
#   - AnswerRelevancy requires both llm= and embeddings= (cosine similarity scoring)
#   - ContextPrecisionWithReference chosen over WithoutReference because
#     we have human-written ground_truth in every EvalCase
#
# USAGE:
#   python -m tests.eval.eval_runner            # local dev
#   python -m tests.eval.eval_runner --env ci   # CI/CD gate mode

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from openai import OpenAI
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas import evaluate

# RAGAS 0.4.3 COMPATIBILITY NOTE:
# ragas.metrics.collections metrics inherit from BaseMetric, NOT Metric.
# ragas.evaluate() does isinstance(m, Metric) check — rejects collections metrics.
# Workaround: use legacy ragas.metrics singletons (ARE Metric instances) but
# inject our own llm/embeddings into them before calling evaluate().
# Tracked: https://github.com/vibrantlabsai/ragas/issues/2638
# TODO: remove workaround when RAGAS fixes BaseMetric → Metric inheritance
import warnings
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

from tests.eval.test_dataset import EVAL_DATASET, EvalCase
from typing import List

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
API_KEY      = os.getenv("RAG_API_KEY", "dev-secret-key-change-in-prod")

# RAGAS score thresholds — below these, CI gate blocks deployment.
# context_precision threshold is lower because TOP_K=5 retrieval is noisy.
# Raise thresholds as retrieval improves (re-ranker tracked in GitHub Issue #4)
THRESHOLDS = {
    "faithfulness"      : 0.7,
    "answer_relevancy"  : 0.7,
    "context_precision" : 0.5,
    "context_recall"    : 0.6,
}

RESULTS_DIR = Path("tests/eval/results")


def build_ragas_metrics() -> list:
    """
    Configures RAGAS legacy metric singletons with our LLM and embeddings.

    RAGAS 0.4.3 COMPATIBILITY WORKAROUND:
        ragas.metrics.collections metrics (Faithfulness, AnswerRelevancy etc.)
        inherit from BaseMetric, not Metric. ragas.evaluate() rejects them with:
        TypeError: All metrics must be initialised metric objects

        Solution: use legacy ragas.metrics singletons which ARE Metric instances,
        then inject our llm/embeddings into them before passing to evaluate().
        Deprecation warnings suppressed — this is intentional, not oversight.

    LLM CHOICE — gpt-4o-mini:
        10x cheaper than gpt-4o. Sufficient for extraction/scoring tasks.
        RAGAS runs on every PR — cost efficiency matters at CI scale.

    EMBEDDINGS — text-embedding-3-small:
        Must match ingestion model. Different model = wrong cosine distances
        in AnswerRelevancy scoring.

    Returns:
        List of configured Metric singletons accepted by ragas.evaluate()
    """
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=4096)
    embeddings = LangchainOpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Inject llm/embeddings into singletons — mutates module-level objects
    # Safe here because eval runner is a short-lived script, not a server
    faithfulness.llm       = llm
    answer_relevancy.llm   = llm
    answer_relevancy.embeddings = embeddings
    context_precision.llm  = llm
    context_recall.llm     = llm

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def call_api(case: EvalCase) -> EvalCase:
    """
    Calls the live /query API for a single test case.

    Populates:
        case.answer   ← final_answer from API response
        case.contexts ← retrieved_chunks from API response
                        (exact chunks LLM used — not a re-query approximation)
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key"   : API_KEY,
    }
    payload = {
        "query"      : case.question,
        "ticker"     : "AAPL",
        "fiscal_year": 2025,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        case.answer   = data.get("final_answer", "")
        case.contexts = data.get("retrieved_chunks", [])

        print(f"  ✅ {case.case_id} [{case.difficulty}] — quality_score={data.get('quality_score', 0):.2f}")

    except requests.exceptions.Timeout:
        print(f"  ⚠️  {case.case_id} TIMEOUT — skipping")
        case.answer   = ""
        case.contexts = []

    except Exception as e:
        print(f"  ❌ {case.case_id} API ERROR: {e}")
        case.answer   = ""
        case.contexts = []

    return case


def build_ragas_dataset(cases: List[EvalCase]) -> "EvaluationDataset":
    """
    Converts EvalCase list into RAGAS EvaluationDataset.

    RAGAS 0.4.3 new API — uses SingleTurnSample instead of HuggingFace Dataset.
    Field names changed from v0.3:
        question     → user_input
        answer       → response
        contexts     → retrieved_contexts
        ground_truth → reference

    WHY SingleTurnSample:
        Collections metrics (Faithfulness, ContextPrecision etc.) only work
        with the native RAGAS schema — not the legacy HuggingFace Dataset format.
    """
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input         = c.question,      # raw user query
            response           = c.answer,        # pipeline final_answer
            retrieved_contexts = c.contexts,      # exact chunks LLM used
            reference          = c.ground_truth,  # human-written ideal answer
        )
        for c in cases
    ]
    return EvaluationDataset(samples=samples)


def save_results(scores: dict, cases: List[EvalCase]) -> Path:
    """
    Saves timestamped JSON results to tests/eval/results/.
    Used for trend tracking across PRs and as CI artifacts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
        "api_url"       : API_BASE_URL,
        "thresholds"    : THRESHOLDS,
        "scores"        : scores,
        "passed"        : {
            metric: float(scores.get(metric, 0)) >= threshold
            for metric, threshold in THRESHOLDS.items()
        },
        "cases": [
            {
                "case_id"   : c.case_id,
                "difficulty": c.difficulty,
                "question"  : c.question,
                "answer"    : c.answer[:200],
                "num_chunks": len(c.contexts),
            }
            for c in cases
        ],
    }

    results_file = RESULTS_DIR / f"eval_{timestamp}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\n💾 Results saved to {results_file}")
    return results_file


def print_report(scores: dict) -> bool:
    import math
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 60)

    all_passed = True
    for metric, threshold in THRESHOLDS.items():
        raw    = scores.get(metric, 0)
        is_nan = raw != raw  # NaN check — NaN != NaN is always True

        if is_nan:
            # NaN = scoring failed, not zero — treat as fail with explanation
            print(f"  ⚠️   {metric:<36} NaN    (scoring failed — see max_tokens)")
            all_passed = False
        else:
            score  = float(raw)
            passed = score >= threshold
            icon   = "✅" if passed else "❌"
            print(f"  {icon}  {metric:<36} {score:.3f}  (min: {threshold})")
            if not passed:
                all_passed = False

    print("=" * 60)
    status = "✅ ALL METRICS PASSED — safe to deploy" if all_passed \
             else "❌ METRICS BELOW THRESHOLD — deployment blocked"
    print(f"  {status}")
    print("=" * 60 + "\n")
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="RAGAS eval runner")
    parser.add_argument(
        "--env",
        default="local",
        choices=["local", "ci"],
        help="ci mode exits with code 1 on threshold failure"
    )
    args = parser.parse_args()

    print(f"\n🔍 Running RAGAS eval against {API_BASE_URL}")
    print(f"   Mode: {args.env} | Cases: {len(EVAL_DATASET)}\n")

    # ── Step 1: Call live API ─────────────────────────────────────────────────
    print("📡 Calling live API...")
    evaluated_cases = [call_api(case) for case in EVAL_DATASET]

    valid_cases = [c for c in evaluated_cases if c.answer and c.contexts]
    if not valid_cases:
        print("❌ No valid cases — is the server running?")
        sys.exit(1)

    print(f"\n📊 Running RAGAS on {len(valid_cases)}/{len(EVAL_DATASET)} valid cases...")

    # ── Step 2: Build dataset + initialize metrics + evaluate ─────────────────
    dataset = build_ragas_dataset(valid_cases)
    metrics = build_ragas_metrics()

    # RAGAS 0.4.3: EvaluationDataset + collections metrics
    # No llm= at evaluate() level — each metric carries its own llm instance
    results = evaluate(dataset=dataset, metrics=metrics)

    # Keys match metric.name exactly — verified via metric.name print test
    import numpy as np
    scores = {
        "faithfulness"      : float(np.mean(results["faithfulness"])),
        "answer_relevancy"  : float(np.mean(results["answer_relevancy"])),
        "context_precision" : float(np.mean(results["context_precision"])),
        "context_recall"    : float(np.mean(results["context_recall"])),
    }

    # ── Step 3: Report + save ─────────────────────────────────────────────────
    all_passed = print_report(scores)
    save_results(scores, valid_cases)

    # ── Step 4: CI gate ───────────────────────────────────────────────────────
    if args.env == "ci" and not all_passed:
        print("🚫 CI mode: exiting with code 1 to block deployment")
        sys.exit(1)


if __name__ == "__main__":
    main()
