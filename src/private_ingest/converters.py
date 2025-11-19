"""Utilities for converting private documents into chunked text blocks."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Protocol


@dataclass
class DocumentChunk:
    """Represents a chunk of text alongside the metadata used for retrieval."""

    content: str
    metadata: Dict[str, str]


class Chunker:
    """Simple sliding window chunker."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> Iterable[str]:
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            yield text[start:end]
            if end == len(text):
                break
            start = end - self.chunk_overlap


class Converter(Protocol):
    def convert(self, path: Path, metadata: Dict[str, str]) -> List[DocumentChunk]:
        """Convert the file at *path* into :class:`DocumentChunk` instances."""


class _MarkdownConverter:
    def __init__(self, chunker: Chunker) -> None:
        self.chunker = chunker

    def convert(self, path: Path, metadata: Dict[str, str]) -> List[DocumentChunk]:
        text = path.read_text(encoding="utf-8")
        return [
            DocumentChunk(content=chunk.strip(), metadata=metadata)
            for chunk in self.chunker.chunk(text)
            if chunk.strip()
        ]


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._text: List[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - trivial
        self._text.append(data)

    def get_text(self) -> str:
        return "".join(self._text)


class _HTMLConverter:
    def __init__(self, chunker: Chunker) -> None:
        self.chunker = chunker

    def convert(self, path: Path, metadata: Dict[str, str]) -> List[DocumentChunk]:
        stripper = _HTMLStripper()
        stripper.feed(path.read_text(encoding="utf-8"))
        text = stripper.get_text()
        return [
            DocumentChunk(content=chunk.strip(), metadata=metadata)
            for chunk in self.chunker.chunk(text)
            if chunk.strip()
        ]


class _PDFConverter:
    def __init__(self, chunker: Chunker) -> None:
        self.chunker = chunker
        try:  # pragma: no cover - optional dependency
            import fitz  # type: ignore

            self._fitz = fitz
        except Exception:  # pragma: no cover - optional dependency
            self._fitz = None

    def convert(self, path: Path, metadata: Dict[str, str]) -> List[DocumentChunk]:
        if self._fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF ingestion. Install it via `pip install pymupdf`."
            )
        doc = self._fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        return [
            DocumentChunk(content=chunk.strip(), metadata=metadata)
            for chunk in self.chunker.chunk(text)
            if chunk.strip()
        ]


def get_converter_for_path(path: Path, chunk_size: int, chunk_overlap: int) -> Converter:
    suffix = path.suffix.lower()
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if suffix in {".md", ".markdown"}:
        return _MarkdownConverter(chunker)
    if suffix in {".html", ".htm"}:
        return _HTMLConverter(chunker)
    if suffix == ".pdf":
        return _PDFConverter(chunker)
    raise ValueError(f"No converter available for extension: {suffix}")
