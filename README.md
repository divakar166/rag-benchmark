# 🔬 RAG Strategy Benchmarker

A production-grade POC for comparing, evaluating, and benchmarking different RAG (Retrieval-Augmented Generation) strategies side by side.

Built with: **FastAPI · Qdrant · OpenAI · Streamlit · RAGAS**

---

## 🏗 Architecture

```
User uploads PDF
       ↓
   PDF Loader (pdfplumber)
       ↓
  Chunking Strategy ──────────────────────────────────────────────┐
  ┌─────────────┬──────────────┬──────────────┬──────────────┐   │
  │ Naive Fixed │   Semantic   │ Hierarchical │    Hybrid    │ HyDE
  └─────────────┴──────────────┴──────────────┴──────────────┘   │
       ↓                                                          │
  OpenAI Embeddings (text-embedding-3-small)                      │
       ↓                                                          │
  Qdrant (separate collection per strategy) ◄─────────────────────┘
       ↓
  User asks a question
       ↓
  Retrieval (dense / hybrid / HyDE)
       ↓
  GPT-4o-mini generates answer
       ↓
  RAGAS evaluation (faithfulness, relevancy, precision, recall)
       ↓
  Streamlit dashboard — side-by-side comparison
```

---

## 🚀 Quick Start

### 1. Clone & setup environment

```bash
git clone https://github.com/yourusername/rag-benchmark
cd rag-benchmark

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
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

## 📡 API Reference

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

## 📁 Project Structure

```
rag-benchmark/
├── core/
│   ├── ingestion/
│   │   ├── chunkers/
│   │   │   ├── base_chunker.py         ← Abstract interface
│   │   │   ├── fixed_chunker.py        ← ✅ Phase 1: Naive baseline
│   │   │   ├── semantic_chunker.py     ← 🔜 Phase 2
│   │   │   └── hierarchical_chunker.py ← 🔜 Phase 2
│   │   ├── pdf_loader.py               ← pdfplumber + pypdf fallback
│   │   ├── embedder.py                 ← OpenAI batched + cached
│   │   └── pipeline.py                 ← Orchestrates full ingestion
│   ├── retrieval/
│   │   ├── dense_retriever.py          ← ✅ Phase 1: cosine search
│   │   ├── hybrid_retriever.py         ← 🔜 Phase 2: BM25 + dense
│   │   └── hyde_retriever.py           ← 🔜 Phase 2: HyDE
│   ├── generation/
│   │   └── generator.py                ← GPT-4o-mini answer generation
│   └── evaluation/
│       └── ragas_evaluator.py          ← 🔜 Phase 3
├── api/
│   └── main.py                         ← FastAPI routes
├── vectordb/
│   └── qdrant_client.py                ← Qdrant wrapper
├── ui/
│   └── app.py                          ← Streamlit dashboard
├── tests/
│   └── ...                             ← 🔜 Phase 3
├── data/
│   ├── uploads/                        ← Uploaded PDFs (gitignored)
│   └── sample_pdfs/                    ← Test PDFs
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config.py
└── .env.example
```

---

## 🧠 RAG Strategies

| #   | Strategy         | Chunking                                  | Retrieval                       | Status     |
| --- | ---------------- | ----------------------------------------- | ------------------------------- | ---------- |
| 1   | **Naive RAG**    | Fixed token windows                       | Dense cosine                    | ✅ Phase 1 |
| 2   | **Semantic**     | Sentence embedding similarity breakpoints | Dense cosine                    | 🔜 Phase 2 |
| 3   | **Hierarchical** | Parent (1024 tok) + Child (256 tok)       | Dense on child, return parent   | 🔜 Phase 2 |
| 4   | **Hybrid**       | Fixed (same as naive)                     | BM25 + Dense via RRF            | 🔜 Phase 2 |
| 5   | **HyDE**         | Fixed (same as naive)                     | Embed hypothetical answer first | 🔜 Phase 2 |

---

## 📊 Evaluation Metrics (Phase 3)

| Metric            | Measures            | Tool   |
| ----------------- | ------------------- | ------ |
| Faithfulness      | Hallucination rate  | RAGAS  |
| Answer Relevancy  | On-topic-ness       | RAGAS  |
| Context Precision | Retrieval precision | RAGAS  |
| Context Recall    | Retrieval recall    | RAGAS  |
| Latency (ms)      | End-to-end speed    | Custom |
| Token Cost        | API cost per query  | Custom |

---

## 🗺 Roadmap

- [x] **Phase 1** — Naive RAG baseline (PDF → Fixed chunks → Qdrant → GPT-4o-mini)
- [ ] **Phase 2** — Semantic, Hierarchical, Hybrid (BM25+Dense), HyDE strategies
- [ ] **Phase 3** — RAGAS evaluation, benchmark runner, results CSV export
- [ ] **Phase 4** — Polish UI, live score charts, deploy to Railway/VPS

---

## 🔑 Environment Variables

| Variable         | Required | Default     | Description                |
| ---------------- | -------- | ----------- | -------------------------- |
| `OPENAI_API_KEY` | ✅       | —           | OpenAI API key             |
| `QDRANT_HOST`    | ❌       | `localhost` | Qdrant host                |
| `QDRANT_PORT`    | ❌       | `6333`      | Qdrant REST port           |
| `CHUNK_SIZE`     | ❌       | `512`       | Tokens per chunk (naive)   |
| `CHUNK_OVERLAP`  | ❌       | `64`        | Overlap tokens             |
| `TOP_K`          | ❌       | `5`         | Chunks retrieved per query |

---

## 🤝 Contributing

PRs welcome. Each new strategy should:

1. Implement `BaseChunker` (if it has a new chunking approach)
2. Register itself in `STRATEGY_REGISTRY` in `core/ingestion/pipeline.py`
3. Add a corresponding Qdrant collection name in `config.py`
