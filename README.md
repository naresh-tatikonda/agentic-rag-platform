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

The system retrieves relevant 10-K chunks using hybrid search, passes them to a multi-agent LangGraph orchestrator, cross-references live market data from Yahoo Finance, scores confidence, and returns a cited, auditable answer.

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
    ├── Market Analyst Agent (Yahoo Finance API)
    └── Critic Node (confidence scoring + rewrite loop)
    │
    ▼
Cost-Aware Router (GPT-4o-mini ↔ GPT-4o)
    │
    ▼
FastAPI Serving Layer (rate limiting + auth middleware)
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
- Full pipeline fires **only on `main` merges** — intentional, not on every commit
- Failed RAGAS gate blocks deployment — EC2 keeps running last good image

---

## Key Engineering Decisions

| Decision | Choice | Why |
|---|---|---|
| Vector Index | HNSW (pgvector) | Sub-millisecond ANN at 1M+ vectors |
| Lexical Search | GIN + ts_rank | BM25-equivalent, no extra extension needed |
| Agent Framework | LangGraph | Stateful graph execution, production-proven |
| LLM Routing | GPT-4o-mini → GPT-4o | Cost control: simple queries use mini |
| Evaluation | RAGAS Faithfulness | Industry standard RAG eval framework |
| Drift Detection | Evidently | Embedding distribution shift monitoring |
| Container Registry | GHCR | Native GitHub integration, free for public repos |

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
| Evaluation | RAGAS, Evidently |
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
