"""FastAPI surface for the private RAG stack."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..private_ingest.pipeline import IngestionPipeline
from ..rag.chatbot import RAGChatbot
from ..rag.embeddings import EmbeddingModel
from ..rag.vector_store import VectorStore


class IngestRequest(BaseModel):
    path: str
    repo: str = "local"
    commit: str = "HEAD"


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4


def _load_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "private_docs.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file at {config_path}")
    return yaml.safe_load(config_path.read_text())


def build_app() -> FastAPI:
    config = _load_config()
    embedding_model = EmbeddingModel(config.get("embedding_model", "hkunlp/instructor-large"))
    vector_store = VectorStore()
    pipeline = IngestionPipeline(config, embedding_model=embedding_model, vector_store=vector_store)
    chatbot = RAGChatbot(vector_store=vector_store, embedding_model=embedding_model)

    app = FastAPI(title="Private Docs RAG")

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict:
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Path {path} does not exist")
        count = pipeline.ingest_path(path, metadata={"repo": request.repo, "commit": request.commit})
        return {"chunks": count}

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict:
        response = chatbot.generate_response(request.question, top_k=request.top_k or 4)
        return {"answer": response.answer, "citations": response.citations}

    return app


app = build_app()
