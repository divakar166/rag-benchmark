"""
FastAPI Application — RAG Benchmark POC (Phase 2)

Endpoints:
    POST /ingest          — Upload a PDF and ingest it for one or all strategies
    POST /query           — Ask a question against one or all strategies
    GET  /collections     — List all Qdrant collections and their stats
    GET  /strategies      — List available strategies
    GET  /health          — Health check
"""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from config import settings
from core.ingestion.pipeline import STRATEGY_REGISTRY, ingest_pdf, ingest_all_strategies
from core.retrieval.dense_retriever import retrieve
from core.retrieval.hybrid_retriever import retrieve_hybrid, invalidate_bm25_cache
from core.retrieval.hyde_retriever import retrieve_hyde
from core.generation.generator import generate_answer
from vectordb.qdrant_client import qdrant_manager


#  App setup 
app = FastAPI(
    title="RAG Strategy Benchmarker",
    description="Compare Naive, Semantic, Hierarchical, Hybrid, and HyDE RAG strategies",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strategies available at query time (includes hierarchical + hyde)
ALL_QUERY_STRATEGIES = ["naive", "semantic", "hierarchical", "hybrid", "hyde"]


#  Request / Response models 
class QueryRequest(BaseModel):
    query: str
    strategy: str = "all"
    top_k: int = settings.top_k


class StrategyResult(BaseModel):
    strategy: str
    answer: str
    chunks_used: int
    latency_ms: float
    sources: list[dict]
    prompt_tokens: int
    completion_tokens: int
    retrieval_scores: list[float]
    extra: dict = {}   # strategy-specific metadata (e.g. HyDE hypothesis)


class QueryResponse(BaseModel):
    query: str
    results: list[StrategyResult]


class EvaluateRequest(BaseModel):
    strategies: list[str] = ["naive", "semantic", "hierarchical", "hybrid", "hyde"]
    top_k: int = settings.top_k
    question_set_path: str = "evaluation/question_set.json"
    output_dir: str = "evaluation/results"


# Routes 
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/strategies")
async def list_strategies():
    return {
        "strategies": ALL_QUERY_STRATEGIES,
        "descriptions": {
            "naive":        "Fixed-size chunking + dense retrieval (baseline)",
            "semantic":     "Semantic boundary chunking via embedding similarity",
            "hierarchical": "Parent-child chunking — small chunks retrieved, large chunks sent to LLM",
            "hybrid":       "BM25 + dense search fused via Reciprocal Rank Fusion",
            "hyde":         "Hypothetical Document Embeddings — LLM generates a hypothesis before searching",
        },
    }


@app.get("/collections")
async def list_collections():
    """Show all Qdrant collections and their stats."""
    all_collections = {
        "naive":                   settings.collection_naive,
        "semantic":                settings.collection_semantic,
        "hierarchical (children)": settings.collection_hierarchical_small,
        "hierarchical (parents)":  settings.collection_hierarchical_large,
        "hybrid":                  settings.collection_hybrid,
        "hyde":                    f"{settings.collection_naive} (shared with naive)",
    }

    results = []
    for label, collection_name in all_collections.items():
        if "(shared" in collection_name:
            results.append({"strategy": label, "note": collection_name})
            continue

        if qdrant_manager.collection_exists(collection_name):
            info = qdrant_manager.get_collection_info(collection_name)
            results.append({"strategy": label, **info})
        else:
            results.append({
                "strategy": label,
                "name": collection_name,
                "status": "not_created",
                "points_count": 0,
            })

    return {"collections": results}


@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    strategy: str = Form(default="all"),
    recreate: bool = Form(default=False),
):
    """
    Upload a PDF and ingest it.
    strategy="all" runs all registered strategies.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_path = Path(settings.upload_dir) / f"{uuid.uuid4()}_{file.filename}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Uploaded file saved to {upload_path}")

    try:
        valid_strategies = list(STRATEGY_REGISTRY.keys()) + ["hierarchical", "all"]
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy '{strategy}'. Choose from: {valid_strategies}"
            )

        if strategy == "all":
            summaries = await ingest_all_strategies(upload_path, recreate=recreate)
        else:
            summaries = [await ingest_pdf(upload_path, strategy=strategy, recreate_collection=recreate)]

        # Invalidate BM25 cache after re-ingestion
        if strategy in ("hybrid", "all"):
            invalidate_bm25_cache(settings.collection_hybrid)

        return {"status": "success", "file": file.filename, "summaries": summaries}

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Ask a question against one or all strategies.
    Returns side-by-side answers with per-strategy metadata.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    strategies_to_run = (
        ALL_QUERY_STRATEGIES
        if request.strategy == "all"
        else [request.strategy]
    )

    for s in strategies_to_run:
        if s not in ALL_QUERY_STRATEGIES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy '{s}'. Available: {ALL_QUERY_STRATEGIES}"
            )

    results: list[StrategyResult] = []

    for strategy in strategies_to_run:
        try:
            result = await _run_strategy(request.query, strategy, request.top_k)
            results.append(result)
        except Exception as e:
            logger.error(f"Strategy '{strategy}' failed: {e}")
            results.append(StrategyResult(
                strategy=strategy,
                answer=f"Error: {str(e)}",
                chunks_used=0,
                latency_ms=0.0,
                sources=[],
                prompt_tokens=0,
                completion_tokens=0,
                retrieval_scores=[],
            ))

    return QueryResponse(query=request.query, results=results)


async def _run_strategy(query: str, strategy: str, top_k: int) -> StrategyResult:
    """Dispatch to the right retriever based on strategy name."""
    extra = {}

    #  Collection readiness check 
    def _not_ingested(strategy_name: str) -> StrategyResult:
        return StrategyResult(
            strategy=strategy_name,
            answer="Not ingested yet. Call POST /ingest first.",
            chunks_used=0, latency_ms=0.0, sources=[],
            prompt_tokens=0, completion_tokens=0, retrieval_scores=[],
        )

    #  Naive 
    if strategy == "naive":
        if not qdrant_manager.collection_exists(settings.collection_naive):
            return _not_ingested(strategy)
        chunks = await retrieve(query, settings.collection_naive, top_k)

    #  Semantic 
    elif strategy == "semantic":
        if not qdrant_manager.collection_exists(settings.collection_semantic):
            return _not_ingested(strategy)
        chunks = await retrieve(query, settings.collection_semantic, top_k)

    #  Hierarchical 
    elif strategy == "hierarchical":
        if not qdrant_manager.collection_exists(settings.collection_hierarchical_small):
            return _not_ingested(strategy)

        # Retrieve children (small, precise)
        child_chunks = await retrieve(query, settings.collection_hierarchical_small, top_k)

        # Fetch their parent chunks (large, context-rich)
        parent_ids = list({
            c["metadata"].get("parent_id")
            for c in child_chunks
            if c["metadata"].get("parent_id") is not None
        })

        if parent_ids:
            parent_chunks = qdrant_manager.fetch_by_ids(
                settings.collection_hierarchical_large, parent_ids
            )
            # Attach retrieval scores from children to parents
            score_map = {
                c["metadata"].get("parent_id"): c["score"] for c in child_chunks
            }
            for p in parent_chunks:
                p["score"] = score_map.get(p.get("chunk_index"), 0.0)
            chunks = parent_chunks
        else:
            chunks = child_chunks   # fallback if parent_id missing

        extra["child_chunks_searched"] = len(child_chunks)
        extra["parent_chunks_fetched"] = len(chunks)

    #  Hybrid 
    elif strategy == "hybrid":
        if not qdrant_manager.collection_exists(settings.collection_hybrid):
            return _not_ingested(strategy)
        chunks = await retrieve_hybrid(query, settings.collection_hybrid, top_k)

    #  HyDE 
    elif strategy == "hyde":
        if not qdrant_manager.collection_exists(settings.collection_naive):
            return _not_ingested(strategy)
        chunks, hyde_meta = await retrieve_hyde(query, settings.collection_naive, top_k)
        extra["hypothesis"] = hyde_meta["hypothesis"]
        extra["hypothesis_latency_ms"] = hyde_meta["hypothesis_latency_ms"]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Generate answer 
    generation = await generate_answer(
        query=query,
        retrieved_chunks=chunks,
        strategy=strategy,
    )

    return StrategyResult(
        strategy=strategy,
        answer=generation["answer"],
        chunks_used=generation["chunks_used"],
        latency_ms=generation["latency_ms"],
        sources=generation["sources"],
        prompt_tokens=generation["prompt_tokens"],
        completion_tokens=generation["completion_tokens"],
        retrieval_scores=[c["score"] for c in chunks],
        extra=extra,
    )


@app.post("/evaluate")
async def evaluate_endpoint(request: EvaluateRequest):
    """
    Run the full RAGAS evaluation benchmark.

    This is a long-running operation (~10-20 min for all strategies × 10 questions).
    Triggers retrieval + generation for every question × strategy combination,
    then scores with RAGAS using NVIDIA NIM as judge.

    Returns a summary with RAGAS scores per strategy + paths to saved results.

    Tip: run with a subset of strategies first to check everything works:
        {"strategies": ["naive", "semantic"], "top_k": 5}
    """
    from core.evaluation.ragas_evaluator import run_evaluation

    question_set = Path(request.question_set_path)
    if not question_set.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Question set not found: {question_set}. "
                   f"Expected at evaluation/question_set.json"
        )

    # Validate strategies
    valid = ["naive", "semantic", "hierarchical", "hybrid", "hyde"]
    for s in request.strategies:
        if s not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy '{s}'. Valid: {valid}"
            )

    logger.info(
        f"Starting evaluation — "
        f"strategies={request.strategies}, top_k={request.top_k}"
    )

    try:
        result = await run_evaluation(
            question_set_path=question_set,
            strategies=request.strategies,
            top_k=request.top_k,
            output_dir=request.output_dir,
        )

        return {
            "status": "success",
            "summary": result["summary"],
            "csv_path": result["csv_path"],
            "json_path": result["json_path"],
            "questions_evaluated": len(result["per_question"]),
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))