"""Retrieval Augmented Generation chatbot implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .embeddings import EmbeddingModel
from .vector_store import SearchResult, VectorStore


@dataclass
class ChatResponse:
    answer: str
    citations: List[Dict[str, str]]


class PromptGenerator:
    """Thin wrapper around the fine-tuned LLM."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class EchoGenerator(PromptGenerator):
    def generate(self, prompt: str) -> str:
        return prompt.split("Assistant:")[-1].strip()


class RAGChatbot:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        generator: Optional[PromptGenerator] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.generator = generator or EchoGenerator()

    def _format_context(self, results: List[SearchResult]) -> List[str]:
        context_blocks = []
        for idx, result in enumerate(results, start=1):
            citation = self._format_citation(idx, result.metadata)
            block = f"[{idx}] ({citation})\n{result.text.strip()}"
            context_blocks.append(block)
        return context_blocks

    def _format_citation(self, idx: int, metadata: Dict[str, str]) -> str:
        file_path = metadata.get("file_path", "unknown")
        commit = metadata.get("commit", "HEAD")
        return f"{file_path}@{commit}"

    def build_prompt(self, question: str, results: List[SearchResult]) -> str:
        context = "\n\n".join(self._format_context(results)) or "No context available."
        return (
            "You are a security-cleared assistant that must cite your sources.\n"
            f"Context:\n{context}\n\n"
            f"User question: {question}\n"
            "Assistant:"
        )

    def generate_response(self, question: str, top_k: int = 4) -> ChatResponse:
        query_embedding = self.embedding_model.embed([question])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k)
        prompt = self.build_prompt(question, results)
        answer = self.generator.generate(prompt)
        citations = [
            {
                "label": f"[{idx}]",
                "file_path": result.metadata.get("file_path", "unknown"),
                "commit": result.metadata.get("commit", "HEAD"),
                "score": result.score,
            }
            for idx, result in enumerate(results, start=1)
        ]
        return ChatResponse(answer=answer, citations=citations)
