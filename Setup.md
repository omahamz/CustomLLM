# Project Setup Guide

Use this guide to reproduce the CustomLLM environment, preprocess data, and
launch the training + retrieval pipelines defined in `src/`.

## 1. Prerequisites
- **OS**: Linux or macOS with Python 3.10+ and Git installed.
- **Hardware**: NVIDIA GPU with at least 40 GB VRAM (A100 80GB recommended) for
  full-size training. CPU-only runs are supported for unit tests and dry runs.
- **CUDA/cuDNN**: Install drivers compatible with the PyTorch build you plan to
  use (see [pytorch.org](https://pytorch.org/get-started/locally/)).

## 2. Clone the repository
```bash
git clone https://example.com/CustomLLM.git
cd CustomLLM
```

## 3. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 4. Install Python dependencies
Install the core stack referenced by the training, ingestion, and evaluation
code.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install accelerate datasets pyarrow sentencepiece pyyaml tqdm
pip install InstructorEmbedding faiss-cpu numpy
pip install pymupdf  # enables PDF ingestion (optional)
```

If you are using different CUDA versions, adjust the first `pip install torch …`
command accordingly.

## 5. Preprocess public datasets
The preprocessing script streams Hugging Face datasets, normalizes whitespace,
and emits sharded Parquet files consumed by training.

```bash
python scripts/preprocess_public.py \
  --dataset bigcode/the-stack-dedup:v1:train:content \
  --dataset openwebtext:train:text \
  --output-dir data/public/processed \
  --shard-size 2000
```

Update the `--dataset` arguments to point at the corpora you are licensed to
use. The resulting shards live under
`data/public/processed/<dataset>/<split>/shard-XXXXX.parquet`.

## 6. Train the tokenizer
Fit the SentencePiece tokenizer across the cleaned corpus. The training script
writes `tokenizer.model`, `tokenizer.vocab`, and metadata to
`artifacts/tokenizer/`.

```bash
python src/tokenization/train_tokenizer.py \
  --input data/public/processed \
  --vocab-size 65536 \
  --output-dir artifacts/tokenizer
```

Point `--input` to either a directory of Parquet shards or raw `.txt` / `.jsonl`
files.

## 7. Configure and launch pretraining
Edit `configs/training/base.yaml` to reference the tokenizer artifact and the
processed shards you produced. Then launch the Accelerate-powered training loop:

```bash
accelerate config  # run once to define your hardware topology
accelerate launch src/training/train.py --config configs/training/base.yaml
```

Checkpoints are written under `artifacts/checkpoints/step-XXXX/` and the final
state saves to `artifacts/checkpoints/final/`.

## 8. Ingest private documents for RAG
Use the ingestion pipeline to chunk private Markdown/HTML/PDF docs, compute
embeddings, and push them into the in-memory vector store. Replace the sample
paths with your own locations.

```python
from pathlib import Path
from src.private_ingest.pipeline import IngestionPipeline

config_path = Path("configs/private_docs.yaml")
pipeline = IngestionPipeline.from_config_file(config_path)
pipeline.ingest_path(Path("/path/to/private/docs"))
pipeline.dump_index(Path("artifacts/vector_store.json"))
```

## 9. Run the RAG chatbot
Instantiate the chatbot with the populated vector store and embedding model.
The default `EchoGenerator` simply returns the prompt tail; swap it with your
fine-tuned model wrapper for real answers.

```python
from src.rag.chatbot import RAGChatbot
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore

vector_store = VectorStore(dim=768)
embedding_model = EmbeddingModel()
chatbot = RAGChatbot(vector_store=vector_store, embedding_model=embedding_model)
response = chatbot.generate_response("Summarize the private docs", top_k=4)
print(response.answer)
print(response.citations)
```

## 10. Run the regression tests
Execute the smoke tests to confirm ingestion and retrieval still align with the
vector store and chatbot APIs.

```bash
pytest tests/rag/test_ingest_retrieval.py
```

## 11. Next steps
- Update `configs/model/base.py` and `configs/training/base.yaml` if you change
  architecture or optimizer hyperparameters.
- Explore `src/evaluation/` for benchmark harnesses (HumanEval, MBPP, doc-grounded QA).
- Review `docs/ops/security.md` for minimum security practices when working with
  private repositories.
