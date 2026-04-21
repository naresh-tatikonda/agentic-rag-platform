# FinSight AI — Agentic RAG Platform for SEC Financial Intelligence

![CI](https://github.com/naresh-tatikonda/agentic-rag-platform/actions/workflows/ci_cd.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green)
![PostgreSQL](https://img.shields.io/badge/pgvector-HNSW%20%2B%20BM25-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production--Serving-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

Production-grade Retrieval-Augmented Generation system that answers natural language questions about public company financials by retrieving and reasoning over SEC 10-K annual filings in real time.

**Example query:**
> "What are Apple's top 3 risk factors for fiscal year 2025 and how did revenue trend?"

The system retrieves relevant 10-K chunks using hybrid search, passes them to a multi-agent LangGraph orchestrator, optionally cross-references live market data from Yahoo Finance (planned), scores confidence, and returns a cited, auditable answer.

## Business Framing

**Problem.** Teams are deploying RAG assistants over sensitive financial and enterprise data, but most of these systems are operated with minimal observability, weak regression testing, and ad‑hoc deployment practices. That leads to silent regressions, inconsistent answer quality, and expensive incidents whenever prompts, models, or data drift.

**Users and impact.** The primary user is the ML platform / MLOps engineer responsible for keeping an LLM assistant over SEC EDGAR filings reliable and cost‑efficient. They need a way to see how retrieval and answer quality evolve over time, detect data and schema drift before it impacts analysts, and gate deployments so only “healthy” versions of the assistant reach production.

**Solution.** FinSight AI wraps an EDGAR-focused RAG pipeline (downloader → parser → chunker → embedder → pgvector + BM25 hybrid retrieval → LangGraph multi‑agent orchestrator → FastAPI API) with CI/CD and observability. The architecture is designed to integrate quality gates (RAG evaluation), dashboards, and future extensions like autoscaling, cost‑aware routing, caching, and drift monitoring so that operating the assistant looks like operating any other critical service: observable, testable, and safe to iterate on.

---

## Architecture

```
EDGAR API
    │
    ▼
Ingestion Pipeline
(Downloader → Parser → Chunker → Embedder)
    │
    ▼
pgvector (HNSW) + BM25 (GIN Index)
    │
    ▼
LangGraph Multi-Agent Orchestrator
    ├── Query Analyzer Node
    ├── SEC Retriever Agent (hybrid search)
    ├── Market Analyst Agent (Yahoo Finance API) [planned]
    └── Critic Node (confidence scoring + rewrite loop)
    │
    ▼
Cost-Aware Router (GPT-4o-mini ↔ GPT-4o) [planned]
    │
    ▼
FastAPI Serving Layer (rate limiting [planned] + auth middleware)
    │
    ▼
Observability Stack
    ├── LangSmith (trace + eval)
    ├── Prometheus (metrics scraping)
    └── Grafana (latency P95, cost/query, confidence dashboards)
    │
    ▼
CI/CD Pipeline (GitHub Actions)
    └── RAGAS Faithfulness Gate (threshold ≥ 0.7) → GHCR → EC2 Deploy
```
> Note: Components marked `[planned]` are architectural extensions, not implemented in the current version.

---

## CI/CD Pipeline

Every merge to `main` runs the full quality gate before deployment:

```
lint (ruff + mypy)
    │
    ▼
pytest (unit + integration)
    │
    ▼
RAGAS Faithfulness Eval (faithfulness ≥ 0.7)
    │  blocked if score < threshold — no deployment
    ▼
Docker Build + Push → GHCR
    │
    ▼
SSH Deploy → EC2
    └── docker compose up -d (zero-downtime rolling restart)
```

- Feature branch pushes run **lint (ruff + mypy) + pytest only** — no RAGAS eval, no deploy
- Full pipeline fires **only on `main` merges that change code or workflow files** — README/docs‑only changes (`*.md`, `docs/**`, `.gitignore`) are skipped via `paths-ignore`.
- Failed RAGAS gate blocks deployment — EC2 keeps running last good image

---

## Key Engineering Decisions

| Decision | Choice | Why |
|---|---|---|
| Vector Index | HNSW (pgvector) | Sub-millisecond ANN at 1M+ vectors |
| Lexical Search | GIN + ts_rank | BM25-equivalent, no extra extension needed |
| Agent Framework | LangGraph | Stateful graph execution, production-proven |
| LLM Routing (planned) | GPT-4o-mini → GPT-4o | Cost control: simple queries use mini |
| Evaluation | RAGAS Faithfulness | Industry standard RAG eval framework |
| Drift Detection (planned) | Evidently | Embedding distribution shift monitoring |
| Container Registry | GHCR | Native GitHub integration, free for public repos |

---

## Failure Modes & Mitigations

Every production system will fail in specific, repeatable ways. This table makes those failure modes explicit across ingestion, retrieval, orchestration, and serving, and documents how the current version of FinSight AI detects and mitigates each one. The roadmap below then focuses on features that improve reliability, visibility, and recovery beyond this baseline.

| # | Failure | Cause | Detection | Mitigation | Status |
|---|--------|-------|-----------|------------|--------|
| 1 | **EDGAR API unavailable** | SEC rate limit or downtime | Downloader raises HTTP error / logs non-2xx response | Ingestion run fails fast instead of silently corrupting state; operator re-runs ingestion once EDGAR is healthy. | Implemented |
| 2 | **Ingestion chunk loss** | Parser fails on malformed 10-K HTML | Mismatch between expected vs inserted rows during local checks | Re-run ingestion for that filing; add unit tests around parser for the pattern that failed. | Partially implemented (manual check; automated row-count guard planned) |
| 3 | **Stale embeddings** | Filing re-issued or amended but old vectors remain | Manual comparison between EDGAR filing date and last ingestion timestamp | Re-ingest and re-embed affected ticker when a newer 10-K is detected. | Planned (design only) |
| 4 | **Retrieval miss** | Query embedding too distant from all stored chunks | Critic confidence < 0.7 after examining retrieved context | Critic retries query rewrite up to 2 times; if confidence still < 0.7, returns “I do not have the answer” instead of hallucinating. | Implemented |
| 5 | **LangGraph node exception** | OpenAI AuthenticationError, timeout, or node logic error | Exception bubbles out of `graph.invoke()` | FastAPI route wraps `graph.invoke()` in try/except; logs error and returns a clean HTTP 500 with a user-friendly “please try again” message. | Implemented |
| 6 | **Critic node infinite loop** | Confidence never reaches 0.7 and loop has no guard | Would show as long-running / hung request | Critic node enforces a hard max of 2 rewrite attempts; after that it exits with a safe “I do not have the answer” response. | Implemented |
| 7 | **OpenAI API timeout / rate limit** | High concurrency or upstream degradation | Timeout / HTTP 429 exception from client | Error propagates to FastAPI layer, which catches it at graph invocation and returns “please try again”; traceback is available in logs for debugging. | Implemented |
| 8 | **RAGAS CI gate failure** | Real quality regression or test-set variance | CI run shows faithfulness < 0.7 for the 10-question suite | Deployment is blocked; engineer inspects traces and fixes regression or, if flaky, adjusts tests before re-running CI. | Implemented |
| 9 | **pgvector slow ANN query** | Missing index or suboptimal HNSW parameters | Observed as high latency during queries / local profiling | Ensure HNSW index is built at ingestion; tune parameters if latency becomes an issue. | Planned (baseline index is present; tuning & alerts planned) |
| 10 | **Docker container crash on EC2** | OOM, bad deploy, or dependency failure | `docker ps` shows container exited; GitHub Actions deploy step fails | `docker compose up -d` can roll back to last good image; failed deploy leaves previous version running. | Implemented (manual monitoring; Prometheus alerting planned) |
| 11 | **Bad API key in production** | Expired or wrong `OPENAI_API_KEY` / `LANGCHAIN_API_KEY` in `.env` | 401 AuthenticationError like the one seen in CI logs | CI fails early; keys get fixed before deployment. Startup-time env validation is planned so bad keys never reach users. | Partially implemented |

---
## Roadmap

Planned production enhancements, ordered by impact on reliability, quality, and operations.

| Tier | # | Feature | Impact |
|------|---|---------|--------|
| 1 | 1 | **Page-Level Citations in Answers** | Make every answer point to the exact 10‑K page/section used, improving analyst trust, auditability, and ease of manual verification. |
| 1 | 2 | **Baseline vs Agentic RAGAS Comparison** | Quantifies how much the LangGraph agentic flow improves over a naive RAG baseline on faithfulness and context recall, grounding future changes in numbers rather than intuition. |
| 1 | 3 | **Caching Layer (Redis / in-memory)** | Reduces latency and OpenAI spend by caching frequent queries and embeddings, keeping the system responsive under load while controlling cost. |
| 1 | 4 | **Cost-Aware LLM Router** | Routes straightforward queries to GPT‑4o‑mini and complex/low‑confidence ones to GPT‑4o, making the quality vs latency vs cost trade‑off explicit and tunable. |
| 2 | 5 | **Cross-Encoder Re-Ranker** | Reranks the top‑k retrieved chunks to improve context precision, reducing irrelevant context and improving answer quality without retraining large models. |
| 2 | 6 | **Yahoo Finance API Integration** | Completes the Market Analyst agent by enriching answers with live market data, introducing real external‑API concerns (timeouts, rate limits, stale data). |
| 2 | 7 | **Adaptive Retrieval Node** | Adjusts retrieval behavior (e.g., re‑query, multi‑hop, or skip) based on critic confidence, improving robustness on ambiguous or underspecified questions. |
| 3 | 8 | **Evidently Drift Monitoring** | Monitors embedding and feature distributions over time, alerting when the data or usage pattern drifts so ingestion, indexing, or prompts can be updated proactively. |
| 3 | 9 | **Kubernetes + HPA** | Runs FastAPI and pgvector on Kubernetes with basic Horizontal Pod Autoscaling, allowing the service to scale with traffic and making deployment closer to typical production stacks. |
| 3 | 10 | **Scaling Strategy Design Doc** | Captures how this architecture would evolve to support larger corpora and higher QPS (sharding, async ingestion, horizontal scaling), providing a blueprint for future growth. |
---

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Python, SEC EDGAR API, BeautifulSoup |
| Vector DB | PostgreSQL + pgvector (HNSW index) |
| Orchestration | LangGraph, LangChain |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Serving | FastAPI, Gunicorn |
| Observability | LangSmith, Prometheus, Grafana |
| Evaluation | RAGAS, Evidently (planned) |
| Deployment | Docker Compose, EC2 |
| CI/CD | GitHub Actions |

---

## Local Development

```bash
# Clone and configure
git clone https://github.com/naresh-tatikonda/agentic-rag-platform
cd agentic-rag-platform
cp .env.example .env        # add your API keys

# Start infrastructure (pgvector, Prometheus, Grafana)
docker compose up -d ragdb

# Fast dev loop — no Docker rebuild needed
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API available at `http://localhost:8000/docs`

---

## Project Structure

```
agentic-rag-platform/
├── app/                    # FastAPI application
│   ├── main.py
│   └── routes/
├── agents/                 # LangGraph multi-agent graph
│   ├── graph.py
│   └── state.py
├── tests/                  # pytest + RAGAS eval suite
│   ├── ragas_eval.py
│   └── fixtures/
├── monitoring/             # Prometheus + Grafana compose
│   ├── docker-compose.yml
│   └── grafana/
├── docker-compose.yml      # Root compose (rag-api + ragdb)
├── Dockerfile
└── .github/workflows/
    └── ci_cd.yml           # Full CI/CD pipeline
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
OPENAI_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=agentic-rag-platform
POSTGRES_DB=ragdb
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://user:pass@localhost:5432/ragdb
API_KEY=
```

---

## Author

**Naresh Tatikonda**

🔗 [LinkedIn](https://www.linkedin.com/in/nareshtatikonda)  
🔗 [GitHub](https://github.com/naresh-tatikonda)
