"""Ingestion helpers for private documents."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Optional dependency: yaml configuration loading
try:  # pragma: no cover - optional
    import yaml
except Exception:  # pragma: no cover - optional
    yaml = None

from .converters import get_converter_for_path
from ..rag.embeddings import EmbeddingModel
from ..rag.vector_store import VectorStore


class IngestionPipeline:
    def __init__(
        self,
        config: Dict,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.config = config
        self.embedding_model = embedding_model or EmbeddingModel(
            config.get("embedding_model", "hkunlp/instructor-large")
        )
        self.vector_store = vector_store or VectorStore(dim=getattr(self.embedding_model, "dimension", 768))
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 64)

    @classmethod
    def from_config_file(cls, path: Path, **kwargs) -> "IngestionPipeline":
        if yaml is None:
            raise ImportError("PyYAML is required to load configuration files")
        config = yaml.safe_load(Path(path).read_text())
        return cls(config=config, **kwargs)

    def ingest_path(self, path: Path, metadata: Optional[Dict[str, str]] = None) -> int:
        path = Path(path)
        metadata = metadata or {}
        if path.is_dir():
            count = 0
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    try:
                        count += self._ingest_file(file_path, metadata)
                    except ValueError:
                        continue
            return count
        return self._ingest_file(path, metadata)

    def _ingest_file(self, path: Path, metadata: Dict[str, str]) -> int:
        converter = get_converter_for_path(path, self.chunk_size, self.chunk_overlap)
        enriched_metadata = self._build_metadata(path, metadata)
        chunks = converter.convert(path, enriched_metadata)
        if not chunks:
            return 0
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = self.embedding_model.embed(texts)
        self.vector_store.add(embeddings, metadatas, texts)
        return len(chunks)

    def _build_metadata(self, path: Path, base_metadata: Dict[str, str]) -> Dict[str, str]:
        metadata = dict(base_metadata)
        metadata.setdefault("file_path", str(path))
        metadata.setdefault("repo", self.config.get("metadata", {}).get("repo", "unknown"))
        metadata.setdefault("commit", self.config.get("metadata", {}).get("commit", "HEAD"))
        metadata.setdefault("source", path.suffix.lstrip("."))
        return metadata

    def dump_index(self, path: Path) -> None:
        serializable = [
            {"text": text, "metadata": metadata}
            for text, metadata in zip(self.vector_store.texts, self.vector_store.metadata)
        ]
        Path(path).write_text(json.dumps(serializable, indent=2))
