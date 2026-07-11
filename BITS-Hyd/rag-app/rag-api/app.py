"""
RAG API service.

Flow:
  1. Embed the incoming question with a local sentence-transformers model.
  2. Search Qdrant for the most relevant chunks.
  3. Build a grounded prompt and send it to vLLM's OpenAI-compatible endpoint,
     which is serving Qwen/Qwen2.5-0.5B-Instruct on the GPU node.
  4. Return the answer + the sources used.
"""

import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

# --- Configuration (all overridable via env vars / ConfigMap) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

app = FastAPI(title="Qwen2.5-0.5B RAG API")

# --- Lazy-loaded singletons ---
_embedder: Optional[SentenceTransformer] = None
_qdrant: Optional[QdrantClient] = None
_llm_client: Optional[OpenAI] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        logger.info(f"Connecting to vLLM OpenAI-compatible endpoint at {VLLM_BASE_URL}")
        # vLLM doesn't require a real key, but the OpenAI SDK insists on a non-empty string
        _llm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    return _llm_client


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class Source(BaseModel):
    text: str
    score: float
    payload: dict


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Checks downstream dependencies are reachable."""
    problems = []
    try:
        get_qdrant().get_collections()
    except Exception as e:
        problems.append(f"qdrant: {e}")
    try:
        get_llm_client().models.list()
    except Exception as e:
        problems.append(f"vllm: {e}")
    if problems:
        raise HTTPException(status_code=503, detail=problems)
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    top_k = req.top_k or TOP_K

    # 1. Embed the question
    embedder = get_embedder()
    query_vector = embedder.encode(req.question).tolist()

    # 2. Retrieve relevant chunks from Qdrant
    qdrant = get_qdrant()
    try:
        hits = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
        ).points
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant query failed: {e}")

    if not hits:
        context_block = "No relevant context was found."
        sources: List[Source] = []
    else:
        context_block = "\n\n".join(
            f"[{i+1}] {h.payload.get('text', '')}" for i, h in enumerate(hits)
        )
        sources = [
            Source(text=h.payload.get("text", ""), score=h.score, payload=h.payload)
            for h in hits
        ]

    # 3. Build the grounded prompt
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the context does not contain the answer, say "
        "you don't have enough information. Cite sources using [1], [2], etc."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {req.question}"

    # 4. Call vLLM (OpenAI-compatible chat completions)
    llm_client = get_llm_client()
    try:
        completion = llm_client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM request failed: {e}")

    answer = completion.choices[0].message.content

    return QueryResponse(answer=answer, sources=sources)
