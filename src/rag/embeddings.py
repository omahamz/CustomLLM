"""Embedding helpers for private RAG workloads."""
from __future__ import annotations

import hashlib
from typing import Callable, Iterable, List, Optional

try:  # pragma: no cover - optional heavy dependency
    from InstructorEmbedding import INSTRUCTOR  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    INSTRUCTOR = None


class EmbeddingModel:
    """Abstraction around the Instructor-large model with graceful fallbacks."""

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        encoder: Optional[Callable[[List[str]], List[List[float]]]] = None,
        dimension: int = 768,
    ) -> None:
        self.model_name = model_name
        self._encoder = encoder
        self._instructor = None
        self.dimension = dimension
        if encoder is None and INSTRUCTOR is not None:  # pragma: no cover - heavy
            self._instructor = INSTRUCTOR(model_name)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        if self._encoder is not None:
            return self._encoder(text_list)
        if self._instructor is not None:  # pragma: no cover - heavy
            instructions = [["Represent the document for retrieval", text] for text in text_list]
            vectors = self._instructor.encode(instructions, show_progress_bar=False)
            return [list(map(float, vector)) for vector in vectors]
        return self._hash_embeddings(text_list)

    def _hash_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Deterministic embedding fallback derived from SHA256 digests."""

        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeated = (digest * ((self.dimension // len(digest)) + 1))[: self.dimension]
            vector = [float(b) for b in repeated]
            norm = sum(x * x for x in vector) ** 0.5 or 1.0
            vectors.append([x / norm for x in vector])
        return vectors
