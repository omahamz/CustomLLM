"""Accelerate-based training loop optimized for CPU-first execution."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.model.base import TransformerConfig
from src.modeling.modeling_custom import CustomTransformer
from src.training.utils.memory import memory_snapshot, near_oom, suggest_batch_size


class StreamingTextDataset(IterableDataset):
    """Stream Parquet shards from disk to avoid loading entire datasets."""

    def __init__(self, files: List[Path], tokenizer_path: Path, seq_length: int, stream_batch_size: int = 256):
        super().__init__()
        self.files = files
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(str(tokenizer_path))
        self.seq_length = seq_length
        self.stream_batch_size = stream_batch_size
        self.pad_id = self.tokenizer.pad_id() if self.tokenizer.pad_id() >= 0 else 0
        self.bos_id = self.tokenizer.bos_id() if self.tokenizer.bos_id() >= 0 else self.pad_id

    def _shard_files(self, worker_id: int, num_workers: int) -> Iterable[Path]:
        if num_workers <= 1:
            return self.files
        return [f for idx, f in enumerate(self.files) if idx % num_workers == worker_id]

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer.encode(text)[: self.seq_length - 1]
        tokens = [self.bos_id] + tokens
        tokens = tokens[: self.seq_length]
        padding = self.seq_length - len(tokens)
        if padding > 0:
            tokens.extend([self.pad_id] * padding)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return input_ids, attention_mask

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        files = self._shard_files(worker_id, num_workers)

        for file in files:
            pf = pq.ParquetFile(file, memory_map=True)
            for batch in pf.iter_batches(columns=["text"], batch_size=self.stream_batch_size):
                column = batch.column(0).to_pylist()
                for raw in column:
                    if not raw:
                        continue
                    input_ids, attention_mask = self._tokenize(str(raw))
                    yield {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_model_config(cfg: Dict, config_path: Path) -> Dict:
    if cfg.get("model"):
        return cfg["model"]
    model_path = cfg.get("model_config")
    if not model_path:
        raise ValueError("Config must contain either a 'model' block or 'model_config' path.")
    path = (config_path.parent / Path(model_path)).resolve()
    return load_yaml(path)


def create_dataloader(training_cfg: Dict, tokenizer_path: Path) -> Tuple[DataLoader, int]:
    files = [Path(p) for p in training_cfg["dataset_paths"]]
    dataset = StreamingTextDataset(files, tokenizer_path, training_cfg["seq_length"], stream_batch_size=training_cfg.get("stream_batch_size", 256))
    num_workers = training_cfg.get("num_workers", 0)
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=False,
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = training_cfg.get("prefetch_factor", 2)
    dataloader = DataLoader(**dataloader_kwargs)
    return dataloader, dataset.pad_id


def rotate_checkpoints(output_dir: Path, max_checkpoints: int) -> None:
    checkpoints = sorted(output_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[-1]))
    while len(checkpoints) > max_checkpoints:
        oldest = checkpoints.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/base.yaml"),
        help="Training configuration YAML.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = TransformerConfig(**resolve_model_config(cfg, args.config))
    training_cfg = cfg["training"]

    cpu_only = not torch.cuda.is_available()
    mixed_precision = training_cfg.get("mixed_precision", "no" if cpu_only else "bf16")
    accelerator = Accelerator(cpu=cpu_only, mixed_precision=mixed_precision)
    if cpu_only:
        torch.set_num_threads(training_cfg.get("cpu_threads", os.cpu_count() or 1))
        accelerator.print("Running in CPU-only mode; enable MKL/BLAS for best performance.")

    if training_cfg.get("auto_scale_batch_size", False):
        adjusted = suggest_batch_size(
            training_cfg["batch_size"],
            training_cfg["seq_length"],
            model_cfg.hidden_size,
            model_cfg.num_layers,
            training_cfg.get("memory_safety_margin", 0.6),
        )
        if adjusted != training_cfg["batch_size"]:
            accelerator.print(
                f"Scaling batch size from {training_cfg['batch_size']} to {adjusted} to fit available RAM."
            )
            training_cfg["batch_size"] = adjusted

    dataloader, pad_token_id = create_dataloader(training_cfg, Path(training_cfg["tokenizer_path"]))
    model = CustomTransformer(model_cfg).to(dtype=model_cfg.torch_dtype())
    if training_cfg.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg.get("weight_decay", 0.01),
        betas=(0.9, 0.95),
    )

    total_steps = training_cfg["num_steps"]
    warmup_steps = training_cfg.get("warmup_steps", 0)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return (current_step + 1) / max(1, warmup_steps)
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if training_cfg.get("resume_from"):
        accelerator.print(f"Resuming from {training_cfg['resume_from']}")
        accelerator.load_state(training_cfg["resume_from"])

    global_step = 0
    model.train()
    output_dir = Path(training_cfg.get("output_dir", "artifacts/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    step_times: List[float] = []
    tokens_per_step = training_cfg["batch_size"] * training_cfg["seq_length"]
    for _ in range(10_000_000):  # large sentinel
        for batch in dataloader:
            step_start = time.perf_counter()
            with accelerator.accumulate(model):
                outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                shift_logits = outputs[:, :-1, :].contiguous()
                shift_labels = batch["input_ids"][:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=pad_token_id,
                )
                accelerator.backward(loss)
                clip_grad_norm_(model.parameters(), training_cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
            if len(step_times) > training_cfg.get("progress_window", 25):
                step_times.pop(0)

            if accelerator.is_main_process and global_step % training_cfg.get("log_every", 10) == 0:
                avg_step = sum(step_times) / max(len(step_times), 1)
                eta_seconds = max(0.0, avg_step * (total_steps - global_step - 1))
                tokens_per_second = tokens_per_step / avg_step if avg_step > 0 else 0.0
                accelerator.print(
                    f"step={global_step} loss={loss.item():.4f} "
                    f"avg_step={avg_step:.2f}s tokens/s={tokens_per_second:.1f} "
                    f"eta={eta_seconds/3600:.2f}h | {memory_snapshot()}"
                )
                if near_oom():
                    accelerator.print(
                        "Warning: system memory is near capacity. Consider lowering batch size or seq_length."
                    )

            global_step += 1
            if accelerator.is_main_process and global_step % training_cfg.get("save_every", 500) == 0:
                checkpoint_path = output_dir / f"step-{global_step}"
                accelerator.save_state(str(checkpoint_path))
                rotate_checkpoints(output_dir, training_cfg.get("max_checkpoints", 3))

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if accelerator.is_main_process:
        final_path = output_dir / "final"
        accelerator.save_state(str(final_path))
        rotate_checkpoints(output_dir, training_cfg.get("max_checkpoints", 3))
        accelerator.print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
