import os
import time
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model choice:
# text-embedding-3-small → 1536 dimensions, $0.02/1M tokens (cost-efficient)
# text-embedding-3-large → 3072 dimensions, $0.13/1M tokens (higher accuracy)
# We use small — sufficient for financial text retrieval at this scale
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS  = 1536


def get_embedding(text: str) -> List[float]:
    """
    Generate a 1536-dimensional embedding vector for a text string.

    The embedding vector encodes semantic meaning — chunks about similar
    topics will have high cosine similarity in vector space.
    This is what enables semantic search beyond keyword matching.

    Args:
        text: Input text (must be under 8191 tokens for this model)

    Returns:
        1536-dimensional embedding vector as List[float]
    """
    text = text.replace("\n", " ").strip()

    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )

    # .data[0].embedding is the 1536-dim vector for our input text
    return response.data[0].embedding


def get_embeddings_batch(chunks: List[dict],
                         batch_size: int = 100) -> List[dict]:
    """
    Embed a list of text chunks in batches using the OpenAI Embeddings API.

    Batching strategy:
    - OpenAI supports up to 2048 inputs per API call
    - batch_size=100 keeps payloads small and avoids timeout errors
    - 0.5s sleep between batches respects the ~3000 RPM rate limit

    Each chunk dict gets an 'embedding' key added containing its
    1536-dim vector — ready for insertion into pgvector.

    Args:
        chunks:     List of chunk dicts from chunker.py
        batch_size: Chunks per API call (default 100)

    Returns:
        Same chunk dicts with 'embedding' vector added to each
    """
    embedded_chunks = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        texts = [chunk["chunk_text"] for chunk in batch]

        print(f"Embedding batch {i // batch_size + 1} "
              f"({len(batch)} chunks, {i + len(batch)}/{total} total)...")

        # Single API call returns one embedding vector per input text
        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )

        # Map each returned vector back to its source chunk
        for chunk, embedding_obj in zip(batch, response.data):
            chunk["embedding"] = embedding_obj.embedding  # 1536-dim vector
            embedded_chunks.append(chunk)

        # Rate limit guard between batches
        if i + batch_size < total:
            time.sleep(0.5)

    return embedded_chunks


if __name__ == "__main__":
    test_chunks = [
        {
            "chunk_text":  "Apple iPhone revenue grew 5% in fiscal 2025.",
            "chunk_index": 0,
            "ticker":      "AAPL",
            "filing_date": "2025-10-31",
            "token_count": 12
        },
        {
            "chunk_text":  "Risk factors include supply chain disruptions.",
            "chunk_index": 1,
            "ticker":      "AAPL",
            "filing_date": "2025-10-31",
            "token_count": 8
        },
        {
            "chunk_text":  "Management expects services revenue to grow.",
            "chunk_index": 2,
            "ticker":      "AAPL",
            "filing_date": "2025-10-31",
            "token_count": 8
        },
    ]

    print("Generating embedding vectors...")
    embedded = get_embeddings_batch(test_chunks, batch_size=100)

    print(f"\nEmbedded {len(embedded)} chunks")
    print(f"Vector dimensions: {len(embedded[0]['embedding'])}")
    print(f"Vector sample (first 5 dims): {embedded[0]['embedding'][:5]}")
