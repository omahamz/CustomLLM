"""Reward-model placeholder used for RLHF experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from configs.model.base import TransformerConfig
from src.modeling.modeling_custom import CustomTransformer


@dataclass
class RewardModelOutput:
    chosen_score: torch.Tensor
    rejected_score: torch.Tensor


class RewardModel(nn.Module):
    """Wrap the base transformer with a scalar reward head."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.transformer = CustomTransformer(config)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer(input_ids, attention_mask)
        values = self.value_head(hidden).squeeze(-1)
        return values

    def score_pair(self, batch: Dict[str, torch.Tensor]) -> RewardModelOutput:
        chosen = self.forward(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rejected = self.forward(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        return RewardModelOutput(chosen[:, -1], rejected[:, -1])
