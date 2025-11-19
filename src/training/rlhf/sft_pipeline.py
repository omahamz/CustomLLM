"""Supervised fine-tuning pipeline for instruction data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm

from configs.model.base import TransformerConfig
from src.modeling.modeling_custom import CustomTransformer


@dataclass
class SFTConfig:
    batch_size: int = 4
    learning_rate: float = 5e-5
    max_steps: int = 10_000


class InstructionDataset(Dataset):
    def __init__(self, records: List[Dict[str, str]], tokenizer_path: str, max_length: int = 1024):
        self.records = records
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        prompt = record["prompt"]
        response = record["response"]
        text = f"<s>Instruction:\n{prompt}\n\nResponse:\n{response}</s>"
        ids = self.tokenizer.encode(text)
        ids = ids[: self.max_length]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SFTPipeline:
    def __init__(self, config: SFTConfig, model_config: TransformerConfig, tokenizer_path: str):
        self.config = config
        self.model = CustomTransformer(model_config)
        self.tokenizer_path = tokenizer_path
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

    def fit(self, dataset: InstructionDataset) -> None:
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self._collate)
        self.model.train()
        for step, batch in enumerate(dataloader):
            outputs = self.model(batch["input_ids"], batch["attention_mask"])
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch["input_ids"].view(-1),
                ignore_index=0,
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if step >= self.config.max_steps:
                break

    @staticmethod
    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}
