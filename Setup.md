# CPU-First Project Setup Guide

Use this guide to reproduce the CustomLLM environment, preprocess data, and
launch training + retrieval pipelines on personal computers without requiring a
GPU. The defaults prioritize CPU stability and memory efficiency; GPU runs will
still work but are no longer required.

## Windows setup with WSL

If you are on Windows, use Windows Subsystem for Linux (WSL) so the dataset
streaming and preprocessing scripts behave like a Linux host. Run these steps
from an elevated PowerShell window:

1. Enable WSL with the default Ubuntu distribution and reboot if prompted:

   ```powershell
   wsl --install -d Ubuntu
   ```

2. Open the newly installed **Ubuntu** app, create your Linux user, and update
   the base packages:

   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3 python3-venv python3-pip git build-essential
   ```

3. Work entirely inside the Ubuntu shell for the rest of this guide. You can
   access your Windows filesystem at `/mnt/c/...` if needed, but cloning the
   repo into your Linux home directory (e.g., `~/CustomLLM`) avoids path quirks.
   All commands below assume you are in the Ubuntu/WSL terminal.

## 1. Prerequisites

- **OS**: Linux or macOS with Python 3.10+ and Git installed. Windows users
  should install [Windows Subsystem for Linux (WSL)](#windows-setup-with-wsl)
  and run all commands from the Linux shell.
  To enter Linux shell, Ctrl + D to exit:
  ```powershell
  Wsl -d Ubuntu
  ```
- **CPU**: Modern x86_64/ARM64 with AVX2 (or newer) and multiple cores for
  parallel data loading. BLAS/MKL-optimized PyTorch wheels are preferred.
- **System RAM** (rough guidance for end-to-end training, including activations
  and dataloader buffers):
  - Tiny (~10M params): **4-8 GB RAM**
  - Small (~50M params): **8-16 GB RAM**
  - Medium (~100M params): **16-32 GB RAM**
- **Disk**: 30-100 GB free for datasets, checkpoints, and tokenizer artifacts.
- **GPU**: Optional. Everything runs on CPU by default.

Typical wall-clock expectations on 8–16 core CPUs:

| Model size     | Batch size | Seq length | Steps | Expected time (CPU) |
| -------------- | ---------- | ---------- | ----- | ------------------- |
| Tiny (~10M)    | 4          | 512        | 5k    | ~8–12 hours         |
| Small (~50M)   | 2          | 512        | 10k   | ~1–2 days           |
| Medium (~100M) | 1          | 1024       | 20k   | ~4–7 days           |

Use these numbers as directional estimates; I/O speed and CPU clocks can shift
results substantially.

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
code. CPU wheels are sufficient; no CUDA toolkit is needed.

```bash
pip install --break-system-packages -r requirements.txt
```

Verify installations (optional):

```bash
python -c "import datasets; print('datasets version:', datasets.__version__)"
python -c "import pyarrow; print('pyarrow version:', pyarrow.__version__)"
python -c "import huggingface_hub; print('huggingface_hub version:', huggingface_hub.__version__)"
```

## 5. Preprocess public datasets

The preprocessing script streams Hugging Face datasets, normalizes whitespace,
and emits sharded Parquet files consumed by the CPU training loop. Keep shard
sizes modest (1–2k rows) to reduce peak memory during preprocessing.

```bash
python3 scripts/preprocess_public.py --dataset bigcode/the-stack-dedup:v1:train:content --dataset openwebtext:train:text --output-dir data/public/processed --shard-size 2000
```

Update the `--dataset` arguments to point at the corpora you are licensed to
use. The resulting shards live under
`data/public/processed/<dataset>/<split>/shard-XXXXX.parquet`.

## 6. Train the tokenizer

Fit the SentencePiece tokenizer across the cleaned corpus. The training script
writes `tokenizer.model`, `tokenizer.vocab`, and metadata to
`artifacts/tokenizer/`.

```bash
python src/tokenization/train_tokenizer.py --input data/public/processed --vocab-size 32768 --output-dir artifacts/tokenizer
```

Point `--input` to either a directory of Parquet shards or raw `.txt` / `.jsonl`
files. Smaller vocabularies (16–32k) reduce memory pressure for CPU runs.

## 7. Configure and launch CPU-first pretraining

Choose a CPU-friendly model preset from `configs/model/` and reference it from a
training config in `configs/training/`:

- `configs/model/cpu_tiny.yaml` – ~10M params, fits in 4–8 GB RAM
- `configs/model/cpu_small.yaml` – ~50M params, fits in 8–16 GB RAM
- `configs/model/cpu_medium.yaml` – ~100M params, fits in 16–32 GB RAM

Example training launch for the tiny preset:

```bash
accelerate launch src/training/train.py --config configs/training/cpu_tiny.yaml
```

If above does not work, it could be that accelerate is NOT in PATH, run:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

The training script will:

- Auto-detect CPU and disable GPU-only code paths.
- Downscale batch size if available RAM is limited.
- Stream Parquet shards from disk to avoid loading the full dataset in memory.
- Apply gradient checkpointing and mixed precision only when compatible.

Checkpoints are written under `artifacts/checkpoints/step-XXXX/` and rotated to
respect CPU RAM limits.

## 8. Manage system resources during long CPU runs

- Close browsers and heavy apps to free memory before starting training.
- Monitor memory usage in the logs; the trainer will warn near OOM thresholds.
- Prefer SSDs over HDDs to keep streaming I/O responsive.
- Pin batch sizes to 1–4 on 8 GB machines; increase gradually after observing
  stable memory usage.
- If training slows due to swapping, lower `training.batch_size` or shorten
  `training.seq_length`.

## 9. Troubleshooting CPU training

- **RuntimeError: out of memory**: Reduce `training.batch_size`, lower
  `training.seq_length`, or switch to the `cpu_tiny` model preset.
- **Slow dataloading**: Lower `training.num_workers` to `0` or `1` to avoid
  contention, and store shards on SSDs.
- **bf16/float16 not supported**: Set `training.mixed_precision: "no"` and keep
  `dtype: float32` in the model configuration.
- **Checkpoints too large**: Increase `training.max_checkpoints` pruning or
  switch to the tiny preset to shrink state size.

## 10. Ingest private documents for RAG (optional)

CPU-friendly ingestion and retrieval continue to work. Replace the sample paths
with your own locations.

```python
from pathlib import Path
from src.private_ingest.pipeline import IngestionPipeline

config_path = Path("configs/private_docs.yaml")
pipeline = IngestionPipeline.from_config_file(config_path)
pipeline.ingest_path(Path("/path/to/private/docs"))
pipeline.dump_index(Path("artifacts/vector_store.json"))
```

## 11. Run the regression tests

Execute the smoke tests to confirm ingestion and retrieval still align with the
vector store and chatbot APIs.

```bash
pytest tests/rag/test_ingest_retrieval.py
```

## 12. Next steps

- Update `configs/model/*.yaml` and `configs/training/*.yaml` when changing
  architecture or optimizer hyperparameters.
- Explore `src/evaluation/` for benchmark harnesses (HumanEval, MBPP,
  doc-grounded QA).
- Review `docs/ops/security.md` for minimum security practices when working with
  private repositories.
