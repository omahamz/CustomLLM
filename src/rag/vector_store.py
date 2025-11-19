"""Minimal vector store with FAISS compatibility."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _np = None

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(vec: Sequence[float]) -> float:
    return sqrt(sum(x * x for x in vec)) or 1.0


def _normalize(vec: Sequence[float]) -> List[float]:
    norm = _norm(vec)
    return [x / norm for x in vec]


@dataclass
class SearchResult:
    text: str
    metadata: Dict[str, str]
    score: float


class VectorStore:
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.texts: List[str] = []
        self.metadata: List[Dict[str, str]] = []
        self._embeddings: List[List[float]] = []
        self._index = None

    def add(self, embeddings: Sequence[Sequence[float]], metadatas: Sequence[Dict[str, str]], texts: Sequence[str]) -> None:
        if not embeddings:
            return
        for embedding, metadata, text in zip(embeddings, metadatas, texts):
            if len(embedding) != self.dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {len(embedding)}")
            self.texts.append(text)
            self.metadata.append(metadata)
            self._embeddings.append(list(map(float, embedding)))
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if faiss is None or _np is None:  # pragma: no cover - optional dependency
            self._index = None
            return
        array = _np.array(self._embeddings, dtype=_np.float32)
        if not len(array):
            self._index = None
            return
        index = faiss.IndexFlatIP(self.dim)
        index.add(array)
        self._index = index

    def search(self, query_embedding: Sequence[Sequence[float]] | Sequence[float], top_k: int = 5) -> List[SearchResult]:
        if not self._embeddings:
            return []
        if isinstance(query_embedding[0], (list, tuple)):
            query = list(map(float, query_embedding[0]))
        else:  # type: ignore[index]
            query = list(map(float, query_embedding))  # type: ignore[arg-type]
        if len(query) != self.dim:
            raise ValueError("Query embedding dimension mismatch")
        if self._index is not None and _np is not None:  # pragma: no cover - optional dependency
            q = _np.array([query], dtype=_np.float32)
            scores, indices = self._index.search(q, top_k)
            order = zip(indices[0], scores[0])
        else:
            q_norm = _normalize(query)
            doc_norms = [_normalize(vec) for vec in self._embeddings]
            similarities = [(_dot(q_norm, doc_vec), idx) for idx, doc_vec in enumerate(doc_norms)]
            order = ((idx, score) for score, idx in sorted(similarities, reverse=True)[:top_k])
        results: List[SearchResult] = []
        for idx, score in order:
            if idx >= len(self.texts):
                continue
            results.append(
                SearchResult(text=self.texts[idx], metadata=self.metadata[idx], score=float(score))
            )
        return results
