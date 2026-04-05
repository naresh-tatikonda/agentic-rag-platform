# agentic-rag-platform

# FinSight AI — Agentic RAG Platform for SEC Financial Intelligence

Multi-agent financial analysis platform processing SEC 10-K filings.
Hybrid semantic + keyword retrieval with pgvector HNSW + BM25,
LangGraph orchestration with cost routing, p99 latency monitoring,
and automated RAG quality evaluation via RAGAS.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green)
![PostgreSQL](https://img.shields.io/badge/pgvector-HNSW%20%2B%20BM25-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Production--Serving-red)
![Kubernetes](https://img.shields.io/badge/Kubernetes-minikube%20%2B%20HPA-blue)

---

## What This System Does

FinSight AI answers natural language questions about public company financials
by retrieving and reasoning over SEC 10-K annual filings in real time.

**Example query:**
> "What are Apple's top 3 risk factors for fiscal year 2025 and how did revenue trend?"

The system retrieves relevant 10-K chunks using hybrid search, passes them to
a multi-agent LangGraph orchestrator, cross-references live market data from
Yahoo Finance, scores confidence, and returns a cited, auditable answer.

---

## Architecture

```
EDGAR API
    |
    v
Ingestion Pipeline
(Downloader -> Parser -> Chunker -> Embedder)
    |
    v
pgvector (HNSW) + BM25 (GIN Index)
    |
    v
LangGraph Multi-Agent Orchestrator
    |-- Query Analyzer Node
    |-- SEC Retriever Agent (hybrid search)
    |-- Market Analyst Agent (Yahoo Finance API)
    +-- Critic Node (confidence scoring + rewrite loop)
    |
    v
Cost-Aware Router (GPT-4o-mini <-> GPT-4o)
    |
    v
FastAPI Serving Layer (rate limiting + auth middleware)
    |
    v
Observability Stack
    |-- LangSmith (trace + eval)
    |-- Prometheus (metrics)
    +-- Grafana (latency P95, cost/query, confidence)
    |
    v
Kubernetes (minikube) + HPA Autoscaling
    |
    v
RAGAS Eval Suite (50Q) + GitHub Actions CI/CD Gate
Evidently Embedding Drift Monitoring
```

## Key Engineering Decisions

| Decision | Choice | Why |
|---|---|---|
| Vector Index | HNSW (pgvector) | Sub-millisecond ANN at 1M+ vectors |
| Lexical Search | GIN + ts_rank | BM25-equivalent, no extra extension needed |
| Agent Framework | LangGraph | Stateful graph execution, production-proven |
| LLM Routing | GPT-4o-mini → GPT-4o | Cost control: simple queries use mini |
| Evaluation | RAGAS | Industry standard RAG eval framework |
| Drift Detection | Evidently | Embedding distribution shift monitoring |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Python, SEC EDGAR API, BeautifulSoup |
| Vector DB | PostgreSQL + pgvector (HNSW index) |
| Orchestration | LangGraph, LangChain |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Serving | FastAPI, Uvicorn |
| Observability | LangSmith, Prometheus, Grafana |
| Evaluation | RAGAS, Evidently |
| Deployment | Docker, Kubernetes (minikube), HPA |
| CI/CD | GitHub Actions |

---

## Build Progress

- [x] Week 1 — SEC EDGAR Ingestion Pipeline
- [ ] Week 2 — LangGraph Multi-Agent Orchestrator
- [ ] Week 3 — FastAPI Serving + Observability
- [ ] Week 4 — Kubernetes + CI/CD + RAGAS Eval

---

## Running Locally

```bash
git clone https://github.com/yourusername/agentic-rag-platform
cd agentic-rag-platform
cp .env.example .env        # Add your OpenAI API key
docker compose up -d        # Start PostgreSQL + pgvector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 src/ingestion/edgar_downloader.py
```

---

## Author
Naresh Tatikonda
