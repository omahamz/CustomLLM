# Model Blueprint

## Architecture Selection
- **Model size**: ~700M-parameter decoder-only transformer with 24 layers, 3072 hidden dimension, 16 attention heads, and an 8k context window. The configuration is sized so a single A100 80GB (or comparable) can host the model with headroom for activation-checkpointing, enabling rapid iteration without distributed inference. Depth/width are tuned to maintain code-aware reasoning (24 layers) while keeping total parameters below the point where pretraining cost would exceed the available budget.
- **Tokenizer**: SentencePiece unigram with a 65k subword vocabulary. The unigram algorithm handles identifier-heavy source files and multilingual documentation by jointly optimizing for frequent programming tokens (e.g., `::`, `camelCase`, `snake_case`) and natural-language words. A 65k vocab keeps embedding matrices manageable while avoiding the fragmentation seen with 32k vocabularies on CJK text and shell commands.
- **Training objectives**: Baseline objective is causal language modeling over contiguous 8k windows. We also reserve capacity for an auxiliary contrastive document-grounding head that scores retrieved passages against generated responses; enabling it during supervised/RAG fine-tuning keeps the base checkpoint unchanged while providing a drop-in module for enterprise deployments that demand grounded citations.

## Pretraining Data Blend
- **70% coding corpora**: Open-source code datasets such as The Stack and StarCoderData (via Hugging Face). We ingest repo metadata to drop GPL/AGPL/LGPL projects, honor removal requests, and exclude binaries/tests that leak secrets. The coding-heavy mix is critical because downstream workflows expect synthesis and refactoring of enterprise SDKs.
- **20% general text**: Broad-coverage corpora including The Pile and SlimPajama ensure the model keeps world knowledge and conversational fluency. These sources are filtered for hate speech and duplicated fragments before mixing to prevent overwhelming the code signal.
- **10% synthetic instruction traces**: Self-instruct and templated dialogues teach the base model multi-turn assistant behavior even before supervised alignment. Synthetic traces are generated with strict prompt hygiene (no customer names, no PHI) to stay within privacy guidelines.
- **Filtering pipeline**: License-aware metadata parsing, automatic GPL/AGPL exclusion, near-duplicate removal (MinHash + SimHash), profanity/PII scrubbing, and language-based bucketing. We store provenance hashes so compliance can reproduce the exact blend or remove offending shards if takedown requests arrive.

## Fine-Tuning Phases
1. **Public alignment SFT**: Fine-tune on instruction datasets such as databricks-dolly-15k, OpenAssistant, and CodeAlpaca. Training emphasizes refusal style, tool-use templates, and step-by-step reasoning so the checkpoint is safe for early internal pilots. Mix in contrastive grounding batches to warm up the auxiliary head without proprietary data.
2. **Private RAG-grounded tuning**: After proprietary documentation, API references, and embedding indices are available, run retrieval-augmented fine-tuning (RA-FT). Prompts include citations and chunk IDs so the model learns to quote sources, disfavor hallucinations, and leverage the optional contrastive head for reranking. Access controls and audit logging are mandatory for this phase.

## Evaluation Targets
- **Code generation**: HumanEval and MBPP pass@1 >= baseline open 7B models. Stretch goal is +5 absolute points over StarCoderBase-1B after pretraining, with further gains from RA-FT.
- **General reasoning**: MMLU subsets covering business, computer security, and software engineering; target ≥60% on relevant subsets to ensure sufficient knowledge transfer from the general-text mix.
- **Chatbot success criteria**: Internal red-teaming should report ≥85% helpful/harmless ratings, <5% unsupported claims in scripted support scenarios, and ≥90% citation coverage when RAG context is present. Customer-ready gating requires a manual evaluation that the contrastive grounding head downranks hallucinated passages for at least 80% of sampled questions.

## Data preprocessing pipeline
1. Install the Hugging Face `datasets` and `pyarrow` dependencies.
2. Stream and clean sources with `scripts/preprocess_public.py`:

   ```bash
   python scripts/preprocess_public.py \
     --dataset bigcode/the-stack-dedup:v1:train:content \
     --dataset openwebtext:train:text \
     --output-dir data/public/processed \
     --shard-size 2000
   ```

   The script normalizes whitespace, annotates fenced code blocks, and emits sharded Parquet under `data/public/processed/<dataset>/<split>/shard-XXXXX.parquet`. The Parquet schema tracks text, number of code blocks, character counts, and any license metadata supplied by the source dataset.

## Tokenizer training
Use `src/tokenization/train_tokenizer.py` to fit a SentencePiece unigram model across a mix of corpora (Parquet shards, `.txt`, or `.jsonl` files). The script materializes a temporary corpus file and writes `tokenizer.model`, `tokenizer.vocab`, and a JSON metadata manifest to `artifacts/tokenizer/`.

```bash
python src/tokenization/train_tokenizer.py \
  --input data/public/processed \
  --vocab-size 65536 \
  --output-dir artifacts/tokenizer
```

## Model architecture
`src/modeling/modeling_custom.py` implements a decoder-only transformer with rotary embeddings, RMSNorm, SiLU feed-forward layers, gradient checkpointing hooks, and tied input/output embeddings. The configuration dataclasses live in `configs/model/base.py` and mirror the planned 700M-parameter setup described above.

## Training (Accelerate + FSDP ready)
- Hyperparameters live in `configs/training/base.yaml`. Update the dataset paths to point at the generated Parquet shards and reference the SentencePiece model.
- Launch the pretraining loop with:

  ```bash
  accelerate launch src/training/train.py --config configs/training/base.yaml
  ```

- The script supports automatic mixed precision, gradient accumulation, gradient checkpointing, warmup/decay learning rate scheduling, and resumable checkpoints (`training.resume_from`). Checkpoints are written to `artifacts/checkpoints/step-XXXX/` using `accelerate.save_state`, making them portable across single-node and multi-node jobs.

### Local vs multi-GPU guidance
- **Local dev**: Run `accelerate config` once, select a single GPU, and keep `batch_size` low (≤2). Use CPU-only dry runs by setting `CUDA_VISIBLE_DEVICES=` and `mixed_precision=fp16`.
- **Multi-GPU clusters**: Configure Accelerate for FSDP or DeepSpeed if your cluster uses sharded training. The training script reads the environment prepared by `accelerate launch`, so there is no extra code to modify. Ensure `data/public/processed` is accessible via shared storage and place checkpoints on the same filesystem for resuming.

## Alignment / RLHF scaffolding
- `src/training/rlhf/reward_model.py`: wraps the base transformer with a scalar value head and utilities to score (chosen, rejected) pairs.
- `src/training/rlhf/ppo_trainer.py`: minimal PPO trainer featuring advantage computation, KL penalties, and policy updates.
- `src/training/rlhf/sft_pipeline.py`: instruction tuning helpers for supervised fine-tuning before PPO. Feed it instruction-response JSONL data plus the SentencePiece tokenizer to bootstrap alignment datasets.

## Evaluation suite
- **HumanEval & MBPP**: `src/evaluation/human_eval.py` runs code generation benchmarks end-to-end (generation + execution). It emits pass@k metrics for HumanEval via the official evaluator and computes accuracy on MBPP by running dataset-provided unit tests.
- **Perplexity dashboards**: `src/evaluation/perplexity.py` streams `.parquet` or text files, feeds them through any Hugging Face causal LM, and reports perplexity along with optional JSON logging for dashboards.
- **Doc-grounded QA**: `src/evaluation/doc_qa.py` consumes JSONL question/context/answer files, generates grounded responses, and measures exact match. Extend this with citation logging by attaching the retrieved document IDs to each record.

## Logging & checkpoint strategy
- Training and evaluation scripts emit structured JSON/STDOUT logs so you can forward them to Weights & Biases (W&B) via `wandb agent` or manual `wandb.init()` calls. Configure `WANDB_MODE=offline` for air-gapped clusters.
- Store tokenizer artifacts in `artifacts/tokenizer/` and checkpoints under `artifacts/checkpoints/step-*`. Long-term storage should periodically rsync these directories to object storage (S3/GCS/Azure Blob) after each epoch while keeping the most recent `n` checkpoints on fast local disks for rapid resumes.
