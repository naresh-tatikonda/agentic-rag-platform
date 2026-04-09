"""
app/metrics.py
--------------
Defines all Prometheus metrics for the Agentic RAG Platform.

WHY THIS FILE EXISTS:
  Centralizing metrics in one module prevents duplicate metric registration
  errors (prometheus_client raises ValueError if you register the same metric
  name twice). Every other module imports from here — they never instantiate
  metrics directly.

METRIC DESIGN DECISIONS:
  - Histogram for latency   → captures full distribution; enables p95/p99 via
                              histogram_quantile() in PromQL / Grafana
  - Gauge for quality_score → value goes up AND down per request; snapshot of
                              the last observed quality score
  - Counter for retries     → always increases; never decreases; tracks
                              cumulative Critic node retry events
  - Counter for requests    → always increases; labeled by intent + status so
                              we can slice traffic by query type and outcome

CARDINALITY SAFETY:
  All labels have bounded, low-cardinality values:
    intent  → 4-6 known values (revenue_summary, risk_analysis, …)
    status  → 2 values (success, error)
  We deliberately exclude ticker, query_text, user_id — those are unbounded
  and would cause cardinality explosion (OOM on Prometheus server).

BUCKET DESIGN (latency):
  Current p50 latency is ~13.4s (3 sequential GPT-4o calls).
  Buckets cover:
    - Low end  (1–5s)   → catches future optimized path (v1.1 parallel nodes)
    - Mid range(8–15s)  → fine-grained resolution around today's SLO boundary
    - High end (18–30s) → catches degraded / timeout cases for alerting
  Bucket selection is permanent — changing them breaks historical time series.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Latency Histogram ────────────────────────────────────────────────────────
# Records the end-to-end wall-clock time for each POST /query request.
# Use histogram_quantile(0.95, rate(rag_request_latency_seconds_bucket[5m]))
# in PromQL to compute rolling p95 latency over a 5-minute window.
RAG_LATENCY = Histogram(
    name="rag_request_latency_seconds",
    documentation="End-to-end latency of each RAG query request in seconds",
    labelnames=["intent"],   # label by intent so we can compare revenue vs risk queries
    buckets=[
        1.0, 2.0, 5.0,          # fast path (future v1.1 optimized)
        8.0, 10.0, 11.0, 12.0,  # approaching current median
        13.0, 14.0, 15.0,       # SLO boundary zone — high resolution here
        18.0, 20.0, 25.0, 30.0, # degraded / near-timeout territory
    ],
)

# ── Quality Score Gauge ──────────────────────────────────────────────────────
# Tracks the most recently observed quality_score from the Critic node (0.0–1.0).
# A Gauge is correct here because quality_score fluctuates per request — it is
# NOT monotonically increasing, so a Counter would be wrong.
# Alert rule example: alert if avg quality_score drops below 0.75 over 10 min.
RAG_QUALITY_SCORE = Gauge(
    name="rag_quality_score",
    documentation="Latest quality score (0.0–1.0) assigned by the Critic node",
    labelnames=["intent"],  # lets us see if specific intents degrade in quality
)

# ── Retry Counter ────────────────────────────────────────────────────────────
# Counts cumulative Critic node retries. A high retry rate signals that the
# Retriever is returning poor chunks OR the LLM is struggling with the prompt.
# Use rate(rag_retry_total[5m]) in PromQL to see retries per second.
RAG_RETRY_TOTAL = Counter(
    name="rag_retry_total",
    documentation="Cumulative number of Critic node retries (quality_score < 0.7 threshold)",
    labelnames=["intent"],
)

# ── Request Counter ──────────────────────────────────────────────────────────
# Counts every POST /query request. Label by intent and status so we can:
#   - See traffic breakdown by query type (capacity planning, cost estimation)
#   - Track error rate: rate(rag_requests_total{status="error"}[5m])
#   - Compute success ratio in Grafana dashboards
RAG_REQUESTS_TOTAL = Counter(
    name="rag_requests_total",
    documentation="Total number of RAG query requests",
    labelnames=["intent", "status"],  # status: "success" | "error"
)
