"""Accelerate-based training loop for the custom transformer."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import yaml
import sentencepiece as spm
import pyarrow.parquet as pq

from configs.model.base import TransformerConfig
from src.modeling.modeling_custom import CustomTransformer


class PackedTextDataset(Dataset):
    """On-the-fly tokenization of text parquet shards."""

    def __init__(self, files: List[Path], tokenizer_path: Path, seq_length: int):
        self.files = files
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(str(tokenizer_path))
        self.seq_length = seq_length
        self.pad_id = self.tokenizer.pad_id()
        if self.pad_id < 0:
            self.pad_id = 0
        self.bos_id = self.tokenizer.bos_id() if self.tokenizer.bos_id() >= 0 else self.pad_id
        self.documents: List[str] = []
        for file in files:
            table = pq.read_table(file, columns=["text"])
            self.documents.extend([str(item) for item in table.column(0).to_pylist() if item])

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer.encode(self.documents[idx])[: self.seq_length]
        tokens = [self.bos_id] + tokens
        tokens = tokens[: self.seq_length]
        padding = self.seq_length - len(tokens)
        if padding > 0:
            tokens.extend([self.pad_id] * padding)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def create_dataloader(training_cfg: Dict) -> Tuple[DataLoader, int]:
    files = [Path(p) for p in training_cfg["dataset_paths"]]
    dataset = PackedTextDataset(files, Path(training_cfg["tokenizer_path"]), training_cfg["seq_length"])
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
    )
    return dataloader, dataset.pad_id


def create_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/base.yaml"),
        help="Training configuration YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = TransformerConfig(**cfg["model"])
    training_cfg = cfg["training"]

    accelerator = Accelerator(mixed_precision=training_cfg.get("mixed_precision", "bf16"))

    dataloader, pad_token_id = create_dataloader(training_cfg)
    model = CustomTransformer(model_cfg)
    if training_cfg.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg.get("weight_decay", 0.01),
        betas=(0.9, 0.95),
    )

    total_steps = training_cfg["num_steps"]
    scheduler = create_scheduler(optimizer, total_steps, training_cfg.get("warmup_steps", 0))

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if training_cfg.get("resume_from"):
        accelerator.print(f"Resuming from {training_cfg['resume_from']}")
        accelerator.load_state(training_cfg["resume_from"])

    global_step = 0
    model.train()
    output_dir = Path(training_cfg.get("output_dir", "artifacts/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(10_000_000):  # large sentinel
        for batch in dataloader:
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

            if accelerator.is_main_process and global_step % training_cfg.get("log_every", 10) == 0:
                accelerator.print(f"step={global_step} loss={loss.item():.4f}")

            global_step += 1
            if accelerator.is_main_process and global_step % training_cfg.get("save_every", 500) == 0:
                checkpoint_path = output_dir / f"step-{global_step}"
                accelerator.save_state(str(checkpoint_path))

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if accelerator.is_main_process:
        final_path = output_dir / "final"
        accelerator.save_state(str(final_path))
        accelerator.print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
