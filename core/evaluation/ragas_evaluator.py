"""
RAGAS Evaluator — Phase 3.

Tested against ragas==0.4.3.

Key API facts discovered through inspection:
  - ascore() takes POSITIONAL STRING ARGS, not a SingleTurnSample object
  - Each metric has a different signature:
      Faithfulness.ascore(user_input, response, retrieved_contexts)
      AnswerRelevancy.ascore(user_input, response)
      ContextPrecision.ascore(user_input, reference, retrieved_contexts)
      ContextRecall.ascore(user_input, retrieved_contexts, reference)
  - ascore() returns MetricResult — access score via .value
  - AnswerRelevancy requires ragas.embeddings.OpenAIEmbeddings (not BaseRagasEmbedding)
  - OpenAIEmbeddings calls client.embeddings.create() — standard OpenAI format
  - We bridge our fastembed HTTP service via a thin FakeEmbeddingsClient adapter
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import math
import httpx
import pandas as pd
from loguru import logger
from openai import AsyncOpenAI

from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from config import settings
from core.retrieval.dense_retriever import retrieve
from core.retrieval.hybrid_retriever import retrieve_hybrid
from core.retrieval.hyde_retriever import retrieve_hyde
from core.generation.generator import generate_answer
from vectordb.qdrant_client import qdrant_manager

class _FakeEmbeddingsResponse:
    """Mimics openai.types.CreateEmbeddingResponse shape that OpenAIEmbeddings expects."""
    def __init__(self, embeddings: list[list[float]]):
        self.data = [type("Embedding", (), {"embedding": e})() for e in embeddings]


class FakeOpenAIEmbeddingsClient:
    """
    Wraps our fastembed HTTP service (/embed) behind the OpenAI embeddings
    interface (client.embeddings.create) that ragas.embeddings.OpenAIEmbeddings uses.

    RAGAS calls: await client.embeddings.create(input=texts, model=model)
    We translate that to: POST embedding_host/embed {"texts": [...]}
    """

    def __init__(self, embedding_host: str, model: str):
        self.embedding_host = embedding_host
        self.model = model
        # RAGAS checks inspect.iscoroutinefunction(client.embeddings.create)
        # so embeddings.create must be an async method on an attribute named 'embeddings'
        self.embeddings = self

    async def create(self, input: str | list[str], model: str = None, **kwargs):
        texts = [input] if isinstance(input, str) else input
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.embedding_host}/embed",
                json={"texts": texts},
                timeout=60.0,
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
        return _FakeEmbeddingsResponse(embeddings)


def _get_ragas_llm():
    """RAGAS 0.4.3: use llm_factory() with AsyncOpenAI client pointed at NVIDIA NIM."""
    nim_client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    return llm_factory(settings.llm_model, client=nim_client)


def _get_ragas_embeddings() -> OpenAIEmbeddings:
    """
    Build OpenAIEmbeddings backed by our fastembed service.
    OpenAIEmbeddings calls client.embeddings.create() — our FakeOpenAIEmbeddingsClient
    translates that to the fastembed /embed endpoint format.
    """
    fake_client = FakeOpenAIEmbeddingsClient(
        embedding_host=settings.embedding_host,
        model=settings.embedding_model,
    )
    return OpenAIEmbeddings(client=fake_client, model=settings.embedding_model)


#  Per-strategy retrieval dispatch 
async def _retrieve_for_strategy(
    query: str,
    strategy: str,
    top_k: int,
) -> list[dict]:
    if strategy == "naive":
        return await retrieve(query, settings.collection_naive, top_k)
    elif strategy == "semantic":
        return await retrieve(query, settings.collection_semantic, top_k)
    elif strategy == "hierarchical":
        child_chunks = await retrieve(query, settings.collection_hierarchical_small, top_k)
        parent_ids = list({
            c["metadata"].get("parent_id")
            for c in child_chunks
            if c["metadata"].get("parent_id") is not None
        })
        if parent_ids:
            parent_chunks = qdrant_manager.fetch_by_ids(
                settings.collection_hierarchical_large, parent_ids
            )
            score_map = {c["metadata"].get("parent_id"): c["score"] for c in child_chunks}
            for p in parent_chunks:
                p["score"] = score_map.get(p.get("chunk_index"), 0.0)
            return parent_chunks
        return child_chunks
    elif strategy == "hybrid":
        return await retrieve_hybrid(query, settings.collection_hybrid, top_k)
    elif strategy == "hyde":
        chunks, _ = await retrieve_hyde(query, settings.collection_naive, top_k)
        return chunks
    raise ValueError(f"Unknown strategy: {strategy}")


#  Question set runner ─
async def _run_strategy_on_question_set(
    strategy: str,
    questions: list[dict],
    top_k: int,
) -> list[dict]:
    rows = []
    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["ground_truth"]
        logger.info(f"  [{strategy}] Q{i+1}/{len(questions)}: {question[:60]}...")

        try:
            chunks = await _retrieve_for_strategy(question, strategy, top_k)
            generation = await generate_answer(
                query=question,
                retrieved_chunks=chunks,
                strategy=strategy,
            )
            rows.append({
                "user_input": question,
                "response": generation["answer"],
                "retrieved_contexts": [c["text"] for c in chunks],
                "reference": ground_truth if isinstance(ground_truth, str)
                             else ground_truth[0] if ground_truth else "",
                "strategy": strategy,
                "latency_ms": generation["latency_ms"],
                "prompt_tokens": generation["prompt_tokens"],
                "completion_tokens": generation["completion_tokens"],
                "chunks_used": generation["chunks_used"],
            })
        except Exception as e:
            logger.error(f"  [{strategy}] Q{i+1} failed: {e}")
            rows.append({
                "user_input": question,
                "response": f"ERROR: {e}",
                "retrieved_contexts": [],
                "reference": ground_truth if isinstance(ground_truth, str)
                             else ground_truth[0] if ground_truth else "",
                "strategy": strategy,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "chunks_used": 0,
            })

        await asyncio.sleep(1.0)

    return rows


# RAGAS scoring
async def _score_with_ragas(rows: list[dict]) -> dict:
    """
    Score rows with RAGAS 0.4.3.

    CRITICAL: ascore() takes POSITIONAL STRING ARGUMENTS, not a SingleTurnSample.
    Each metric has a different signature — call them individually.
    ascore() returns MetricResult — access the float via .value
    """
    logger.info(f"Starting RAGAS scoring for {len(rows)} total rows")

    # Filter valid rows
    valid_rows = [
        r for r in rows
        if r.get("response") 
        and not str(r.get("response", "")).startswith("ERROR")
        and r.get("retrieved_contexts")  # must have contexts
    ]

    skipped = len(rows) - len(valid_rows)
    if skipped > 0:
        logger.warning(f"Skipped {skipped} rows due to missing response, ERROR response, or no retrieved contexts")

    if not valid_rows:
        logger.warning("No valid rows to score with RAGAS")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None
        }
    
    logger.info(f"Proceeding with {len(valid_rows)} valid rows for RAGAS evaluation")

    # Initialize RAGAS components
    try:
        ragas_llm = _get_ragas_llm()
        ragas_embeddings = _get_ragas_embeddings()
        logger.debug("RAGAS LLM and embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAGAS LLM or embeddings: {e}", exc_info=True)
        raise

    # Initialize metrics
    faithfulness_metric   = Faithfulness(llm=ragas_llm)
    answer_rel_metric     = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    context_prec_metric   = ContextPrecision(llm=ragas_llm)
    context_recall_metric = ContextRecall(llm=ragas_llm)

    logger.info("RAGAS metrics initialized. Starting evaluation...")

    async def _safe_score(coro, metric_name: str, row_idx: int) -> float:
        try:
            result = await coro
            score = float(result.value)
            logger.debug(f"  [{metric_name}] Row {row_idx}: score = {score:.4f}")
            return score
        except Exception as e:
            logger.warning(f"  [{metric_name}] Row {row_idx} failed: {e}")
            return float("nan")

    # Scoring each metric
    n = len(valid_rows)

    logger.info(f"Computing Faithfulness on {n} samples...")
    faith_scores = await asyncio.gather(*[
        _safe_score(
            faithfulness_metric.ascore(
                r["user_input"], r["response"], r["retrieved_contexts"]
            ),
            "faithfulness", i
        ) for i, r in enumerate(valid_rows)
    ])

    logger.info(f"Computing Answer Relevancy on {n} samples...")
    ans_rel_scores = await asyncio.gather(*[
        _safe_score(
            answer_rel_metric.ascore(r["user_input"], r["response"]),
            "answer_relevancy", i
        ) for i, r in enumerate(valid_rows)
    ])

    logger.info(f"Computing Context Precision on {n} samples...")
    ctx_prec_scores = await asyncio.gather(*[
        _safe_score(
            context_prec_metric.ascore(
                r["user_input"], r["reference"], r["retrieved_contexts"]
            ),
            "context_precision", i
        ) for i, r in enumerate(valid_rows)
    ])

    logger.info(f"Computing Context Recall on {n} samples...")
    ctx_recall_scores = await asyncio.gather(*[
        _safe_score(
            context_recall_metric.ascore(
                r["user_input"], r["retrieved_contexts"], r["reference"]
            ),
            "context_recall", i
        ) for i, r in enumerate(valid_rows)
    ])

    def _avg(scores: list[float]) -> float | None:
        valid = [s for s in scores if not (isinstance(s, float) and math.isnan(s))]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 4)

    results = {
        "faithfulness":      _avg(faith_scores),
        "answer_relevancy":  _avg(ans_rel_scores),
        "context_precision": _avg(ctx_prec_scores),
        "context_recall":    _avg(ctx_recall_scores),
    }

    # Final summary logging
    logger.info("RAGAS scoring completed. Results:")
    for name, val in results.items():
        if val is not None:
            logger.info(f"  • {name:18}: {val:.4f}")
        else:
            logger.warning(f"  • {name:18}: None (all scores failed)")

    # log number of successful scores per metric
    for name, scores in [
        ("faithfulness", faith_scores),
        ("answer_relevancy", ans_rel_scores),
        ("context_precision", ctx_prec_scores),
        ("context_recall", ctx_recall_scores)
    ]:
        valid_count = sum(1 for s in scores if not (isinstance(s, float) and math.isnan(s)))
        logger.debug(f"  {name}: {valid_count}/{n} successful scores")

    return results


# Public API 
async def run_evaluation(
    question_set_path: str | Path,
    strategies: Optional[list[str]] = None,
    top_k: int = settings.top_k,
    output_dir: str | Path = "core/evaluation/results",
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(question_set_path) as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions from {question_set_path}")

    all_strategies = list(strategies or ["naive", "semantic", "hierarchical", "hybrid", "hyde"])

    # Verify collections exist
    for s in list(all_strategies):
        if s == "hyde":
            collection = settings.collection_naive
        elif s == "hierarchical":
            collection = settings.collection_hierarchical_small
        else:
            collection = getattr(settings, f"collection_{s}", None)
        if collection and not qdrant_manager.collection_exists(collection):
            logger.warning(f"Strategy '{s}' collection not found — skipping")
            all_strategies.remove(s)

    all_rows: list[dict] = []
    strategy_summaries: dict[str, dict] = {}

    for strategy in all_strategies:
        logger.info(f"Evaluating strategy: {strategy.upper()}")

        start = time.perf_counter()
        rows = await _run_strategy_on_question_set(strategy, questions, top_k)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(f"Scoring {strategy} with RAGAS...")
        ragas_scores = await _score_with_ragas(rows)

        valid_rows = [r for r in rows if not r["response"].startswith("ERROR")]
        avg_latency = sum(r["latency_ms"] for r in valid_rows) / max(len(valid_rows), 1)
        avg_tokens = sum(
            r["prompt_tokens"] + r["completion_tokens"] for r in valid_rows
        ) / max(len(valid_rows), 1)

        strategy_summaries[strategy] = {
            "ragas": ragas_scores,
            "latency_avg_ms": round(avg_latency, 2),
            "tokens_avg": round(avg_tokens, 2),
            "questions_evaluated": len(valid_rows),
            "total_eval_time_ms": round(elapsed, 2),
        }
        all_rows.extend(rows)

        logger.info(f"{strategy} RAGAS scores: {ragas_scores}")
        logger.info(f"{strategy} avg latency: {avg_latency:.0f}ms")

        await asyncio.sleep(3.0)

    # Save outputs 
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(all_rows)
    csv_path = output_path / f"eval_per_question_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Per-question results saved to {csv_path}")

    summary = {
        "timestamp": timestamp,
        "strategies": strategy_summaries,
        "question_set": str(question_set_path),
        "top_k": top_k,
    }
    json_path = output_path / f"eval_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {json_path}")

    return {
        "summary": summary,
        "per_question": all_rows,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }