"""
Generator — takes retrieved chunks + query → calls LLM → returns answer.

The prompt is engineered for RAG:
  - Instructs the model to stay grounded in the provided context
  - Explicitly says "I don't know" if the context doesn't cover the question
  - Returns structured output including source attribution
"""
import time
from loguru import logger
from openai import AsyncOpenAI
from config import settings

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError

_client = AsyncOpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)

# Prompt template
SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say exactly:
"I don't have enough information in the provided context to answer this question."

Rules:
- Be concise and factual
- Do not add information from your training data
- If you cite specific facts, mention which source/page they came from
- Format your answer in clear, readable prose
"""


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source_info = f"[Source: {chunk.get('source', 'unknown')}, Page: {chunk.get('page', '?')}]"
        lines.append(f"--- Context {i} {source_info} ---\n{chunk['text']}")
    return "\n\n".join(lines)


async def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    strategy: str = "unknown",
) -> dict:
    """
    Generate an answer given a query and retrieved chunks.

    Returns:
        {
            "answer": str,
            "strategy": str,
            "chunks_used": int,
            "latency_ms": float,
            "sources": list[dict],   ← chunk source + page + score
            "prompt_tokens": int,
            "completion_tokens": int,
        }
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant context was retrieved for this query.",
            "strategy": strategy,
            "chunks_used": 0,
            "latency_ms": 0.0,
            "sources": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    
    
    try:
        await _client.models.list()
    except Exception as e:
        logger.warning(f"Warm-up call failed: {e}")

    context_block = _build_context_block(retrieved_chunks)

    user_message = f"""CONTEXT:
{context_block}

QUESTION: {query}

ANSWER:"""

    logger.debug(f"Generating answer for strategy='{strategy}', chunks={len(retrieved_chunks)}")

    start = time.perf_counter()
    response = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    answer = response.choices[0].message.content.strip()
    usage = response.usage

    sources = [
        {
            "source": c.get("source"),
            "page": c.get("page"),
            "score": round(c.get("score", 0.0), 4),
            "chunk_index": c.get("chunk_index"),
        }
        for c in retrieved_chunks
    ]

    logger.info(
        f"Generated answer — strategy={strategy}, "
        f"latency={latency_ms:.0f}ms, "
        f"tokens={usage.prompt_tokens}+{usage.completion_tokens}"
    )

    return {
        "answer": answer,
        "strategy": strategy,
        "chunks_used": len(retrieved_chunks),
        "latency_ms": round(latency_ms, 2),
        "sources": sources,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }
