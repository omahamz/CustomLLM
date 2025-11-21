"""Command-line entrypoint for asking the RAG chatbot a question."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.private_ingest.pipeline import IngestionPipeline
from src.rag.chatbot import RAGChatbot
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


DEFAULT_CONFIG = Path("configs/private_docs.yaml")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    return yaml.safe_load(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question to the private RAG chatbot.")
    parser.add_argument(
        "question",
        help="The user question to pass to the chatbot.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        help="Optional file or directory to ingest before asking the question.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the ingestion configuration. Defaults to {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        dest="top_k",
        help="Number of context chunks to retrieve for the answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    embedding_model = EmbeddingModel(config.get("embedding_model", "hkunlp/instructor-large"))
    vector_store = VectorStore(dim=getattr(embedding_model, "dimension", 768))
    pipeline = IngestionPipeline(config, embedding_model=embedding_model, vector_store=vector_store)
    chatbot = RAGChatbot(vector_store=vector_store, embedding_model=embedding_model)

    if args.docs is not None:
        if not args.docs.exists():
            raise FileNotFoundError(f"Provided docs path does not exist: {args.docs}")
        metadata = config.get("metadata", {})
        ingested = pipeline.ingest_path(args.docs, metadata=metadata)
        print(f"Ingested {ingested} chunks from {args.docs}")

    response = chatbot.generate_response(args.question, top_k=args.top_k)

    print("Answer:\n" + response.answer)
    if response.citations:
        print("\nCitations:")
        for citation in response.citations:
            label = citation.get("label", "[?]")
            file_path = citation.get("file_path", "unknown")
            commit = citation.get("commit", "HEAD")
            score = citation.get("score", 0.0)
            print(f"{label} {file_path}@{commit} (score={score:.3f})")
    else:
        print("\nNo citations available.")


if __name__ == "__main__":
    main()