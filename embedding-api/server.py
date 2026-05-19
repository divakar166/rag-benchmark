import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding

os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

app = FastAPI()

_model: Optional[TextEmbedding] = None


class EmbeddingRequest(BaseModel):
    texts: list[str]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": _model is not None,
    }


def get_model() -> TextEmbedding:
    global _model

    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)

    return _model


@app.post("/embed")
def embed(req: EmbeddingRequest):
    if not req.texts:
        return {
            "model": MODEL_NAME,
            "dimension": 0,
            "embeddings": [],
        }

    try:
        model = get_model()
        vectors = [vec.tolist() for vec in model.embed(req.texts)]

        return {
            "model": MODEL_NAME,
            "dimension": len(vectors[0]) if vectors else 0,
            "embeddings": vectors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))