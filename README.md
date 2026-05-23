# RAG Strategy Benchmarker

A production-grade POC for comparing, evaluating, and benchmarking different RAG (Retrieval-Augmented Generation) strategies side by side.

Built with: **FastAPI · Qdrant · OpenAI / Custom LLM (vLLM) · Streamlit · RAGAS**

---

## Architecture

```
User uploads PDF
       ↓
   PDF Loader (pdfplumber)
       ↓
  Chunking Strategy ──────────────────────────────────────────────┐
  ┌─────────────┬──────────────┬──────────────┬──────────────┐    │
  │ Naive Fixed │   Semantic   │ Hierarchical │    Hybrid    │   HyDE
  └─────────────┴──────────────┴──────────────┴──────────────┘    │
       ↓                                                          │
   Embeddings (BAAI/bge-small-en-v1.5 / OpenAI)                   │
       ↓                                                          │
  Qdrant (separate collection per strategy) ◄─────────────────────┘
       ↓
  User asks a question
       ↓
  Retrieval (dense / hybrid / HyDE)
       ↓
  Modular LLM Client (OpenAI GPT / Custom vLLM / Qwen) generates answer
       ↓
  RAGAS evaluation (faithfulness, relevancy, precision, recall)
       ↓
  Streamlit dashboard
```

---

## Quick Start

### 1. Clone & setup environment

```bash
git clone https://github.com/divakar166/rag-benchmark.git
cd rag-benchmark

uv sync

cp .env.example .env
# Edit .env and set your LLM_PROVIDER to "openai" or "custom"
```

### 2. Start Qdrant with Docker

```bash
# Just Qdrant (recommended during development)
docker compose up qdrant -d

# Verify it's running
curl http://localhost:6333/healthz
# → {"title":"qdrant - vector search engine"}
```

### 3. Start the FastAPI backend

```bash
uvicorn api.main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### 4. Start the Streamlit UI

```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

---

## Docker Compose Setup

The project ships with a `docker-compose.yml` that orchestrates three services:

| Service           | Container           | Port(s)        | Description                                                             |
| ----------------- | ------------------- | -------------- | ----------------------------------------------------------------------- |
| **qdrant**        | `rag_qdrant`        | `6333`, `6334` | Qdrant vector database (REST + gRPC)                                    |
| **embedding-api** | `rag_embedding_api` | `8001`         | Lightweight FastAPI server running fastembed (`BAAI/bge-small-en-v1.5`) |
| **api**           | `rag_api`           | `8000`         | Main FastAPI backend (ingestion, retrieval, generation, evaluation)     |

### Run the full stack

```bash
# Start all services (Qdrant + Embedding API + FastAPI backend)
docker compose up -d

# Verify everything is healthy
curl http://localhost:6333/healthz   # Qdrant
curl http://localhost:8001/health    # Embedding API
curl http://localhost:8000/health    # FastAPI backend
```

### Run individual services

```bash
# Only Qdrant (recommended during local development)
docker compose up qdrant -d

# Qdrant + Embedding API (run backend locally with uvicorn)
docker compose up qdrant embedding-api -d
```

### Persistent volumes

| Volume            | Purpose                            |
| ----------------- | ---------------------------------- |
| `qdrant_storage`  | Qdrant collections & vector data   |
| `embedding_cache` | Downloaded embedding model weights |

### Rebuild after code changes

```bash
docker compose up -d --build
```

---

## API Reference

| Method | Endpoint       | Description                  |
| ------ | -------------- | ---------------------------- |
| `GET`  | `/health`      | Health check                 |
| `GET`  | `/strategies`  | List available strategies    |
| `GET`  | `/collections` | Qdrant collection stats      |
| `POST` | `/ingest`      | Upload PDF for ingestion     |
| `POST` | `/query`       | Ask question, get comparison |

### Example: Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_paper.pdf" \
  -F "strategy=all" \
  -F "recreate=false"
```

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main contributions of this paper?",
    "strategy": "all",
    "top_k": 5
  }'
```

---

## Project Structure

```
rag-benchmark/
├── core/                                ← Core RAG logic
│   ├── ingestion/
│   │   ├── chunkers/
│   │   │   ├── base_chunker.py          ← Abstract chunker interface
│   │   │   ├── fixed_chunker.py         ← Naive fixed-window baseline
│   │   │   ├── semantic_chunker.py      ← Embedding-similarity breakpoints
│   │   │   └── hierarchical_chunker.py  ← Parent / child chunking
│   │   ├── pdf_loader.py               ← pdfplumber + pypdf fallback
│   │   ├── embedder.py                 ← HTTP client for embedding-api
│   │   └── pipeline.py                 ← Orchestrates full ingestion
│   ├── retrieval/
│   │   ├── dense_retriever.py           ← Dense cosine search
│   │   ├── hybrid_retriever.py          ← BM25 + dense via RRF
│   │   └── hyde_retriever.py            ← Hypothetical Document Embeddings
│   ├── generation/
│   │   ├── llm_client.py               ← Modular LLM client factory
│   │   └── generator.py                ← LLM answer generation
│   └── evaluation/
│       ├── ragas_evaluator.py           ← RAGAS metric evaluation
│       ├── benchmark_runner.py          ← Batch benchmark runner
│       ├── question_set.json            ← Predefined evaluation questions
│       └── results/                     ← CSV & JSON evaluation outputs
├── api/
│   └── main.py                          ← FastAPI routes
├── embedding-api/                       ← Standalone embedding microservice
│   ├── server.py                        ← FastAPI server (fastembed)
│   ├── Dockerfile
│   └── requirements.txt
├── vectordb/
│   └── qdrant_client.py                 ← Qdrant wrapper
├── ui/
│   └── app.py                           ← Streamlit dashboard
├── tests/
│   └── ...
├── data/
│   └── uploads/                         ← Uploaded PDFs (gitignored)
├── modal_vllm.py                        ← Modal cloud vLLM deployment
├── docker-compose.yml                   ← Multi-service Docker setup
├── Dockerfile                           ← API service image
├── config.py                            ← Pydantic settings & defaults
├── pyproject.toml
├── requirements.txt
├── requirements-api.txt                 ← Slim deps for Docker API image
└── .env.example
```

---

## RAG Strategies

| #   | Strategy         | Chunking                                  | Retrieval                       |
| --- | ---------------- | ----------------------------------------- | ------------------------------- |
| 1   | **Naive RAG**    | Fixed token windows                       | Dense cosine                    |
| 2   | **Semantic**     | Sentence embedding similarity breakpoints | Dense cosine                    |
| 3   | **Hierarchical** | Parent (1024 tok) + Child (256 tok)       | Dense on child, return parent   |
| 4   | **Hybrid**       | Fixed (same as naive)                     | BM25 + Dense via RRF            |
| 5   | **HyDE**         | Fixed (same as naive)                     | Embed hypothetical answer first |

---

## Evaluation Metrics

| Metric            | Measures            | Tool   |
| ----------------- | ------------------- | ------ |
| Faithfulness      | Hallucination rate  | RAGAS  |
| Answer Relevancy  | On-topic-ness       | RAGAS  |
| Context Precision | Retrieval precision | RAGAS  |
| Context Recall    | Retrieval recall    | RAGAS  |
| Latency (ms)      | End-to-end speed    | Custom |
| Token Cost        | API cost per query  | Custom |

---

## Environment Variables

| Variable          | Default                          | Description                                              |
| ----------------- | -------------------------------- | -------------------------------------------------------- |
| `LLM_PROVIDER`    | `custom`                         | LLM client provider (`custom` or `openai`)               |
| `LLM_MODEL`       | `Qwen/Qwen2.5-Coder-7B-Instruct` | The LLM model identifier                                 |
| `OPENAI_API_KEY`  | —                                | Standard OpenAI API Key (when `LLM_PROVIDER=openai`)     |
| `LLM_API_KEY`     | `dummy`                          | API Key for custom self-hosted LLM                       |
| `LLM_BASE_URL`    | `http://localhost:8002/v1`       | Base URL for custom OpenAI-compatible LLM endpoint       |
| `EMBEDDING_HOST`  | `http://localhost:8001`          | Embedding API base URL                                   |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5`         | Embedding model name                                     |
| `QDRANT_HOST`     | `localhost`                      | Qdrant database host                                     |
| `QDRANT_PORT`     | `6333`                           | Qdrant REST port                                         |
| `HF_TOKEN`        | —                                | Hugging Face token (optional, for gated model downloads) |
| `CHUNK_SIZE`      | `512`                            | Tokens per chunk (naive)                                 |
| `CHUNK_OVERLAP`   | `64`                             | Overlap tokens                                           |
| `TOP_K`           | `5`                              | Chunks retrieved per query                               |

---

## Contributing

PRs welcome. Each new strategy should:

1. Implement `BaseChunker` (if it has a new chunking approach)
2. Register itself in `STRATEGY_REGISTRY` in `core/ingestion/pipeline.py`
3. Add a corresponding Qdrant collection name in `config.py`
