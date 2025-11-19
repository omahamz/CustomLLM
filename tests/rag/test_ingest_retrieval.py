from src.private_ingest.pipeline import IngestionPipeline
from src.rag.chatbot import RAGChatbot
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


def keyword_encoder(texts):
    keywords = ["security", "compliance", "pipeline"]
    embeddings = []
    for text in texts:
        lowered = text.lower()
        row = [float(lowered.count(keyword)) for keyword in keywords]
        embeddings.append(row)
    return embeddings


def test_ingest_and_retrieve(tmp_path):
    config = {"chunk_size": 64, "chunk_overlap": 8, "metadata": {"repo": "test", "commit": "abc123"}}
    embedding_model = EmbeddingModel(encoder=keyword_encoder, dimension=3)
    vector_store = VectorStore(dim=3)
    pipeline = IngestionPipeline(config, embedding_model=embedding_model, vector_store=vector_store)

    doc_path = tmp_path / "security.md"
    doc_path.write_text("Security compliance pipeline\nThis line mentions security.")

    chunks = pipeline.ingest_path(doc_path)
    assert chunks > 0

    chatbot = RAGChatbot(vector_store=vector_store, embedding_model=embedding_model)
    response = chatbot.generate_response("What does the security pipeline mention?", top_k=1)

    assert response.citations, "Expected at least one citation"
    assert response.citations[0]["file_path"].endswith("security.md")
