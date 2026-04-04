from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


def count_tokens(text: str) -> int:
    """
    Count exact token count using OpenAI's cl100k_base tokenizer.
    Used to verify chunk sizes stay within embedding model limits.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def chunk_text(text: str, ticker: str, filing_date: str,
               chunk_size: int = 512, overlap: int = 50) -> List[dict]:
    """
    Split 10-K text into semantically meaningful overlapping chunks.

    Strategy: RecursiveCharacterTextSplitter with tiktoken length function.

    RecursiveCharacterTextSplitter tries to split in this priority order:
      1. Double newline (paragraph boundary)  — best split point
      2. Single newline (line boundary)        — second best
      3. Period + space (sentence boundary)    — third best
      4. Single space (word boundary)          — last resort
      5. Character level                       — absolute fallback

    This ensures chunks end at natural language boundaries rather than
    mid-sentence, which preserves semantic meaning for better retrieval.

    Combining with tiktoken as the length function ensures chunk_size=512
    means exactly 512 OpenAI tokens — aligning perfectly with the
    text-embedding-3-small model's input space.

    Args:
        text:        Clean plain text from parser.py
        ticker:      Stock ticker (e.g. 'AAPL') for metadata tagging
        filing_date: Date string (e.g. '2025-10-31') for filtering
        chunk_size:  Max tokens per chunk (default 512)
        overlap:     Token overlap between chunks (default 50)

    Returns:
        List of chunk dicts with text, metadata, and token count
    """
    # Use tiktoken as the length function so chunk_size is measured
    # in tokens (not characters) — critical for embedding model alignment
    enc = tiktoken.get_encoding("cl100k_base")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=lambda t: len(enc.encode(t)),  # token-based sizing
        separators=["\n\n", "\n", ". ", " ", ""]        # priority order
    )

    # Split the text into chunks respecting natural boundaries
    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text_content in enumerate(raw_chunks):
        token_count = count_tokens(chunk_text_content)

        # Skip very short chunks — likely headers or artifacts
        if token_count > 20:
            chunks.append({
                "chunk_text":  chunk_text_content,
                "chunk_index": i,
                "ticker":      ticker,
                "filing_date": filing_date,
                "token_count": token_count
            })

    return chunks


if __name__ == "__main__":
    sample_text = """
    Apple Inc. designs, manufactures, and markets smartphones, personal computers,
    tablets, wearables, and accessories worldwide. The Company also sells various
    related services. The Company's products include iPhone, Mac, iPad, and
    Wearables, Home and Accessories. Services include advertising, AppleCare,
    cloud services, digital content, and payment services. Apple was founded in
    1976 and is headquartered in Cupertino, California.
    """ * 50

    chunks = chunk_text(sample_text, ticker="AAPL", filing_date="2025-10-31")

    print(f"Total chunks: {len(chunks)}")
    print(f"\nChunk 1 ({chunks[0]['token_count']} tokens):")
    print(chunks[0]['chunk_text'][:300])
    print(f"\nChunk 2 ({chunks[1]['token_count']} tokens):")
    print(chunks[1]['chunk_text'][:300])
