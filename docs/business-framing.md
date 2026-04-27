# FinSight AI — Problem Statement and Solution Approach

## Problem Statement

### Context

Financial analysts and engineering teams increasingly rely on LLM‑powered assistants to answer questions over SEC 10‑K filings and other regulatory documents. These assistants are often deployed as prototypes: they can answer simple questions in demos, but they rarely come with the observability, testing, and deployment discipline needed for reliable use in day‑to‑day analysis.

In the context of SEC filings, this gap is especially risky. A hallucinated risk factor, an incorrect revenue figure, or a stale data point is not just a UX issue; it can lead to wrong decisions, erode trust in the system, and raise compliance concerns. Analysts need answers that are both *useful* and *verifiable* down to the exact place in the filing.

### Concrete problems this project targets

This project focuses on a specific set of production problems that appear when you deploy a RAG system over SEC 10‑K filings:

1. **No reliable way to detect quality regressions.**  
   When prompts, model versions, or ingestion logic change, answer quality can silently degrade. Most teams ship updates without automated checks that compare new behavior to a baseline, so regressions are discovered only after users complain.

2. **Limited visibility into retrieval quality.**  
   Traditional monitoring tells you if the API is up and how fast it responds, but not whether the retrieval step is returning relevant context. A request that returns HTTP 200 may still be built on irrelevant or low‑quality chunks, leading to hallucinated answers.

3. **Data and embedding drift without alerts.**  
   Over time, the distribution of queries and filings changes: companies restructure their reports, new terminology appears, and embedding models are updated. Without drift monitoring, retrieval and generation quality can degrade even if the code has not changed.

4. **Ad‑hoc deployment decisions.**  
   New versions of the system are often deployed manually, based on a few spot‑checks or intuition. There is no explicit “quality gate” in the CI/CD pipeline that decides whether a change is acceptable based on objective metrics.

5. **Undefined responses to common failure modes.**  
   In real systems, many things can go wrong: EDGAR API downtime, parsing failures, retrieval misses, LangGraph node exceptions, OpenAI rate limits, bad API keys, or container crashes. When these are not enumerated and tied to concrete mitigations, incidents are slow to resolve and often recur in the same way.

The goal of FinSight AI is to address these problems directly: not just “answer questions about 10‑K filings,” but do so in a way that is observable, testable, and deployable as a production service.

---

## Solution Approach

### High‑level design

FinSight AI is an end‑to‑end Retrieval‑Augmented Generation (RAG) system for SEC 10‑K filings with three core properties:

- It provides a **natural language interface** over EDGAR filings, returning answers that are grounded in retrieved text and can be traced back to specific sections.
- It uses a **hybrid retrieval stack** (pgvector HNSW + BM25) orchestrated through LangGraph to combine dense semantic search with exact keyword matching.
- It wraps the RAG core in **production concerns**: FastAPI serving, CI/CD with a RAGAS-based quality gate, and observability via Prometheus, Grafana, and LangSmith.

The design is intentionally modular: ingestion, retrieval, orchestration, serving, and monitoring are separated so each can be evolved or replaced without rewriting the whole system.

### Ingestion and storage

1. **Data source.**  
   The system ingests 10‑K filings from the SEC EDGAR API. Each filing is downloaded as HTML and parsed into structured text.

2. **Parsing and chunking.**  
   HTML is cleaned and split into chunks (e.g., by section headers and paragraph boundaries). The chunk size is chosen to balance two needs:
   - Small enough that each chunk is topically focused.
   - Large enough that a single chunk usually contains enough context to answer a specific question without stitching many chunks together.

3. **Embedding and storage.**  
   Each chunk is embedded using an OpenAI embedding model and stored in PostgreSQL with the pgvector extension. Two indexing strategies are used:
   - A **vector index (HNSW)** for approximate nearest neighbor search over embeddings.
   - A **text index (GIN + `tsvector`)** to provide BM25‑like keyword search.

This design keeps everything in a single, well‑understood database (PostgreSQL) while still enabling modern vector search.

### Hybrid retrieval

When a user sends a question, the retrieval layer performs **hybrid search**:

- **Dense retrieval (HNSW)** finds chunks that are semantically similar to the query.
- **Sparse retrieval (BM25)** finds chunks that match important keywords and phrases exactly.

The results from both searches are combined and ranked before being passed to the orchestration layer. This ensures that:

- Conceptual questions (e.g., “what are the key risk factors?”) benefit from embeddings.
- Precise questions (e.g., “what was revenue for fiscal 2024?”) still get exact matches.

Future improvements like Reciprocal Rank Fusion (RRF) and cross‑encoder reranking are planned to refine this merging, but the current design already establishes the hybrid pattern.

### Agentic orchestration with LangGraph

The orchestration layer is built in LangGraph and implements **explicit decision logic**, not just a linear “call model → retry if low confidence” loop.

The main nodes are:

1. **Query Analyzer Node**  
   Classifies the query into types such as:
   - Factual/retrieval (“What are Apple’s 2024 risk factors?”)
   - Comparative (“How did Apple’s revenue change from 2023 to 2024?”)
   - Market‑enriched (future, with Yahoo Finance)

   This classification determines which downstream agents engage and how many chunks to retrieve.

2. **SEC Retriever Agent**  
   Executes hybrid retrieval using pgvector and BM25, assembles the top‑k chunks, and passes them along with the query to the critic.

3. **Critic Node**  
   Evaluates whether the retrieved context is sufficient to answer the question. It uses a structured prompt to produce a confidence score and justification. The behavior is:

   - If confidence ≥ 0.7: proceed to answer generation.
   - If confidence < 0.7 and fewer than 2 attempts have been made: rewrite the query (e.g., clarify ticker/year), re‑run retrieval, and re‑evaluate.
   - If confidence < 0.7 after 2 attempts: return an explicit “I do not have enough information to answer this” message instead of fabricating an answer.

   This is a bounded decision loop: the agent knows when to stop and how to fail safely.

4. **(Planned) Market Analyst Agent and Cost‑Aware Router**  
   - A Market Analyst agent will add live market context (e.g., price reaction around earnings) via Yahoo Finance.  
   - A cost‑aware router will choose between GPT‑4o‑mini and GPT‑4o based on query complexity and required confidence, making the quality vs cost trade‑off explicit.

Even in its current form, this orchestration captures the core “agentic” behavior: classify, choose tools, evaluate confidence, retry with structured rewrites, and decide when to abstain.

### Serving and API layer

The orchestrated RAG flow is exposed through a **FastAPI** service:

- Requests hit FastAPI endpoints that validate input and call the LangGraph graph.
- The call to `graph.invoke()` is wrapped in `try/except` so that model errors, timeouts, or node logic errors result in a controlled HTTP error response rather than an unhandled exception.
- The service is containerized and run under Gunicorn/Uvicorn, making it easy to deploy with Docker Compose and (in future) Kubernetes.

This layer is where external clients integrate: internal tools, UIs, or other services can call the API with a question and receive structured, cited answers.

### Observability and monitoring

FinSight AI includes observability at three levels:

- **LangSmith** traces each node in the LangGraph graph, capturing inputs, outputs, and latencies. This makes it possible to see, for example, how often the Critic triggers rewrites or safe exits.
- **Prometheus** scrapes metrics from the FastAPI service: request counts, latency histograms, error rates, and custom metrics such as confidence scores.
- **Grafana** dashboards visualize these metrics (latency percentiles, error rate, cost per query proxy) so that engineers can quickly see whether the system is healthy and whether recent changes have impacted performance.

These tools turn the system from a black box into something operators can reason about.

### CI/CD and quality gate

The project includes a **GitHub Actions** pipeline that enforces a basic quality gate before deployment:

1. Static checks (`ruff`, `mypy`).
2. Unit and integration tests (`pytest`).
3. **RAGAS‑based faithfulness evaluation** on a small, curated test set. If the faithfulness score drops below a threshold (e.g., 0.7), the pipeline fails and deployment is blocked.
4. Docker image build and push to GitHub Container Registry.
5. Deploy to EC2 via SSH and `docker compose up -d`.

Documentation‑only changes (e.g., README and `docs/` updates) are excluded via `paths-ignore`, so the full pipeline only runs when code or configuration changes.

---

## How this design addresses the original problem

- **Silent regressions** are mitigated by the RAGAS gate in CI/CD and LangSmith traces that show how behavior changed between versions.
- **Retrieval observability** improves through hybrid search metrics, Critic confidence logging, and Grafana dashboards showing latency and error rates.
- **Drift** can be monitored over time (and will be reinforced with Evidently in the roadmap) by watching how RAGAS scores and retrieval distributions change on a fixed evaluation set.
- **Ad‑hoc deploys** are replaced with a repeatable pipeline where deployments only proceed when tests and evaluation pass.
- **Failure modes** are cataloged with defined detection and mitigation strategies, so common issues (API downtime, rate limits, bad keys) have predictable outcomes instead of surprising users.

This combination of architecture and operational practices turns FinSight AI from a one‑off demo into a system that can realistically be run and evolved in a production environment.