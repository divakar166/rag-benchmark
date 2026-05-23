"""
HyDE Retriever (Hypothetical Document Embeddings)

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)

How it works:
  1. Take the user's query
  2. Ask the LLM: "Write a short passage that would answer this question"
  3. Embed the HYPOTHETICAL PASSAGE
  4. Search Qdrant with that embedding
  5. Return the real chunks nearest to the hypothesis

Why it works:
  - Queries and documents live in different embedding spaces
    ("What is X?" ≠ "X is a concept that...")
  - A hypothetical answer lives in DOCUMENT space — same distribution
    as the real chunks — so similarity search is much more effective
  - Works especially well for definitions, explanations, multi-hop questions

Why it can fail:
  - If the LLM hallucinates a wrong hypothesis, you retrieve garbage
  - Adds one extra LLM call per query (latency cost)
  - Short factual queries ("What year was X founded?") gain little from HyDE

Architecture note:
  - Reuses the SAME Qdrant collection as naive (settings.collection_naive)
  - Only the query vector changes — chunking and storage are identical
  - The hypothesis is never stored, only used to produce a search vector
  - main.py exposes the hypothesis text in StrategyResult.extra so you
    can inspect what the LLM hypothesised for each query
"""
import time
from loguru import logger

from config import settings
from core.ingestion.embedder import embed_query
from vectordb.qdrant_client import qdrant_manager
from core.generation.llm_client import get_llm_client

_client = get_llm_client()

# HyDE prompt 
HYDE_SYSTEM_PROMPT = """You are a document writing assistant.
Your task is to write a SHORT factual passage (3-5 sentences) that would
directly answer the given question.

Rules:
- Write as if you are an excerpt from a relevant document or paper
- Be specific and use technical language appropriate to the domain
- Do NOT say "I don't know" — always write a plausible passage
- Do NOT reference the question itself in your answer
- Output ONLY the passage, no preamble or explanation
"""


async def _generate_hypothesis(query: str) -> tuple[str, float]:
    """
    Ask the LLM to write a hypothetical document passage for the query.
    Returns (hypothesis_text, latency_ms).

    Uses temperature=0.5 (slightly higher than generation) so the hypothesis
    covers more vocabulary and bridges the query-document gap better.
    """
    start = time.perf_counter()

    response = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}"},
        ],
        max_tokens=256,
        temperature=0.5,
        top_p=settings.top_p,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    hypothesis = response.choices[0].message.content.strip()

    logger.debug(f"HyDE hypothesis ({latency_ms:.0f}ms): {hypothesis[:120]}...")
    return hypothesis, latency_ms


async def retrieve_hyde(
    query: str,
    collection_name: str,
    top_k: int = settings.top_k,
    score_threshold: float = settings.score_threshold,
) -> tuple[list[dict], dict]:
    """
    HyDE retrieval: generate hypothesis → embed it → search Qdrant.

    Returns:
        chunks   : same shape as dense_retriever.retrieve()
        metadata : {"hypothesis": str, "hypothesis_latency_ms": float}
                   surfaced in StrategyResult.extra by main.py
    """
    logger.debug(f"HyDE retrieve — query='{query[:60]}...'")

    # Step 1: generate a hypothetical answer passage
    hypothesis, hypothesis_latency_ms = await _generate_hypothesis(query)

    # Step 2: embed the hypothesis
    hypothesis_vector = await embed_query(hypothesis)

    # Step 3: search Qdrant with the hypothesis vector
    results = qdrant_manager.search(
        collection_name=collection_name,
        query_vector=hypothesis_vector,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    metadata = {
        "hypothesis": hypothesis,
        "hypothesis_latency_ms": round(hypothesis_latency_ms, 2),
    }

    logger.debug(
        f"HyDE — hypothesis_latency={hypothesis_latency_ms:.0f}ms, "
        f"retrieved={len(results)} chunks"
    )

    return results, metadata