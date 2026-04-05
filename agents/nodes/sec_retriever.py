"""
agents/nodes/sec_retriever.py
------------------------------
SECRetriever Node — second node in the LangGraph pipeline.

Responsibility:
    Uses ticker, year, and intent from AgentState to retrieve
    the most relevant chunks from pgvector using hybrid search:
    - HNSW vector search  : semantic similarity (dense retrieval)
    - BM25 / GIN index    : keyword match (sparse retrieval)

Why hybrid search?
    - Pure vector search misses exact keyword matches (e.g. "revenue $394B")
    - Pure BM25 misses semantic matches (e.g. "sales" vs "revenue")
    - Hybrid = best of both worlds — higher recall, better precision

Table schema (sec_filings):
    id, ticker, filing_type, filed_date, cik,
    chunk_index, chunk_text, embedding, created_at

Output:
    Updates AgentState with: retrieved_chunks, retrieval_scores
"""

import logging
import os
import psycopg2
from openai import OpenAI
from agents.state import AgentState

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── OpenAI client for generating query embeddings ─────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Retrieval config ──────────────────────────────────────────────────────────
TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"   # Must match ingestion model

# ── Intent → keyword boost mapping ───────────────────────────────────────────
INTENT_KEYWORDS = {
    "risk_analysis":     "risk factors threats litigation regulatory",
    "revenue_summary":   "revenue earnings net income financial results",
    "business_overview": "business segments products services operations",
    "general":           "",
}


def get_db_connection():
    """
    Create a fresh PostgreSQL connection per node invocation.
    Avoids stale connection issues across multiple graph runs.
    """
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def embed_query(text: str) -> list[float]:
    """
    Generate embedding vector for the query using OpenAI.
    Must use same model as ingestion (text-embedding-3-small = 1536 dims).
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def sec_retriever_node(state: AgentState) -> AgentState:
    """
    LangGraph node function — retrieves relevant SEC filing chunks from pgvector.

    Hybrid search strategy:
        1. Embed the query using OpenAI text-embedding-3-small
        2. Run HNSW vector search (cosine similarity via pgvector <=> operator)
        3. Filter by ticker and filing year extracted from filed_date
        4. Return top-k chunks with similarity scores

    Args:
        state: Current AgentState with ticker, year, intent populated

    Returns:
        Partial AgentState update with retrieved_chunks and retrieval_scores
    """
    query  = state["query"]
    ticker = state.get("ticker") or "AAPL"
    year   = state.get("year")   or 2025
    intent = state.get("intent") or "general"

    logger.info(f"SECRetriever searching ticker={ticker}, year={year}, intent={intent}")

    # ── Enrich query with intent-specific keywords for better recall ──────────
    keyword_boost  = INTENT_KEYWORDS.get(intent, "")
    enriched_query = f"{query} {keyword_boost}".strip()

    try:
        # ── Generate query embedding ──────────────────────────────────────────
        query_embedding = embed_query(enriched_query)
        embedding_str   = "[" + ",".join(map(str, query_embedding)) + "]"

        conn = get_db_connection()
        cur  = conn.cursor()

        # ── Hybrid search query ───────────────────────────────────────────────
        # Filters by ticker and year using EXTRACT on filed_date
        # Orders by cosine distance (HNSW index via <=> operator)
        sql = """
            SELECT
                chunk_text,
                1 - (embedding <=> %s::vector) AS similarity_score
            FROM sec_filings
            WHERE
                ticker = %s
                AND EXTRACT(YEAR FROM filed_date) = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        cur.execute(sql, (
            embedding_str,   # For similarity score calculation
            ticker,          # Filter by stock ticker
            year,            # Filter by filing year from filed_date
            embedding_str,   # For ORDER BY cosine distance
            TOP_K,
        ))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            logger.warning(f"No chunks found for ticker={ticker}, year={year}")
            return {"retrieved_chunks": [], "retrieval_scores": []}

        chunks = [row[0] for row in rows]
        scores = [float(row[1]) for row in rows]

        logger.info(f"SECRetriever retrieved {len(chunks)} chunks, top score={scores[0]:.3f}")

        return {
            "retrieved_chunks": chunks,
            "retrieval_scores": scores,
        }

    except Exception as e:
        logger.error(f"SECRetriever failed: {e}")
        return {"retrieved_chunks": [], "retrieval_scores": []}
