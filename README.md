# CustomLLM CPU-First Blueprint

This repository hosts a CPU-friendly language model pipeline spanning data preprocessing, tokenizer training, model training, retrieval-augmented ingestion, and evaluation. GPU runs still work, but the defaults prioritize stability and memory efficiency on modern multi-core CPUs.

## Quickstart (mirrors [Setup.md](./Setup.md))
1. Clone the repo and enter it:
   ```bash
   git clone https://example.com/CustomLLM.git
   cd CustomLLM
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
3. Install dependencies (CPU wheels are sufficient):
   ```bash
   pip install -r requirements.txt
   ```
4. Preprocess public datasets into Parquet shards:
   ```bash
   python scripts/preprocess_public.py \
     --dataset bigcode/the-stack-dedup:v1:train:content \
     --dataset openwebtext:train:text \
     --output-dir data/public/processed \
     --shard-size 2000
   ```
5. Train the SentencePiece tokenizer:
   ```bash
   python src/tokenization/train_tokenizer.py \
     --input data/public/processed \
     --vocab-size 32768 \
     --output-dir artifacts/tokenizer
   ```
6. Launch a CPU-first training run (choose a preset in `configs/model/`):
   ```bash
   accelerate launch src/training/train.py --config configs/training/cpu_tiny.yaml
   ```
7. Smoke-test RAG ingestion and retrieval:
   ```bash
   pytest tests/rag/test_ingest_retrieval.py
   ```

Windows users should follow the [WSL instructions](./Setup.md#windows-setup-with-wsl) and run all commands from the Ubuntu shell.

## Architecture snapshot
- **Model size presets**: CPU-optimized configs for ~10M, ~50M, and ~100M parameter decoder-only transformers live in `configs/model/` (`cpu_tiny.yaml`, `cpu_small.yaml`, `cpu_medium.yaml`).
- **Tokenizer**: SentencePiece unigram with configurable vocabulary (16–32k recommended for CPU memory limits).
- **Training objectives**: Causal LM with optional contrastive document-grounding head for RAG-aligned fine-tuning.

## Data preprocessing
`scripts/preprocess_public.py` streams Hugging Face datasets, normalizes whitespace, and emits sharded Parquet files under `data/public/processed/<dataset>/<split>/shard-XXXXX.parquet`. Keep shard sizes modest (1–2k rows) to reduce peak memory during preprocessing.

## Tokenizer training
Use `src/tokenization/train_tokenizer.py` to fit a SentencePiece unigram model across Parquet shards, `.txt`, or `.jsonl` files. Outputs (`tokenizer.model`, `tokenizer.vocab`, metadata) are written to `artifacts/tokenizer/`.

## Training
`src/modeling/modeling_custom.py` implements the decoder-only transformer. Training hyperparameters live in `configs/training/`—pair them with the CPU presets for RAM-aware batch sizes and sequence lengths.

Example launch:
```bash
accelerate launch src/training/train.py --config configs/training/cpu_tiny.yaml
```

The trainer auto-detects CPUs, streams Parquet shards from disk, and checkpoints under `artifacts/checkpoints/step-XXXX/`. Adjust `training.batch_size` and `training.seq_length` if you approach OOM on smaller machines.

## Retrieval-augmented ingestion (optional)
Use `src/private_ingest/pipeline.py` with `configs/private_docs.yaml` to index private documents and dump a vector store:
```python
from pathlib import Path
from src.private_ingest.pipeline import IngestionPipeline

config_path = Path("configs/private_docs.yaml")
pipeline = IngestionPipeline.from_config_file(config_path)
pipeline.ingest_path(Path("/path/to/private/docs"))
pipeline.dump_index(Path("artifacts/vector_store.json"))
```

## Evaluation targets
- **Code generation**: HumanEval and MBPP pass@1 with competitive baselines.
- **General reasoning**: MMLU subsets focused on business/security/software knowledge.
- **Chatbot behavior**: High helpful/harmless ratings with grounded citations when RAG context is available.

## Operations and security
Consult `docs/ops/security.md` for handling private data and `Setup.md` for detailed CPU-first guidance, troubleshooting tips, and resource-management recommendations.
