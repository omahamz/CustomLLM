"""Simplified PPO trainer scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from src.training.rlhf.reward_model import RewardModel


@dataclass
class PPOConfig:
    learning_rate: float = 1e-5
    clip_range: float = 0.2
    vf_coef: float = 0.1
    kl_coef: float = 0.1
    batch_size: int = 2
    ppo_epochs: int = 4


class PPOTrainer:
    def __init__(self, actor: torch.nn.Module, reward_model: RewardModel, config: PPOConfig):
        self.actor = actor
        self.reward_model = reward_model
        self.config = config
        self.optimizer = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate)

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return rewards - values

    def step(self, rollouts: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        total_loss = 0.0
        for _ in range(self.config.ppo_epochs):
            for batch in rollouts:
                logits = self.actor(batch["input_ids"], attention_mask=batch["attention_mask"])
                log_probs = F.log_softmax(logits, dim=-1)
                chosen = log_probs.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    rewards = self.reward_model(batch["input_ids"], batch["attention_mask"])
                advantages = self.compute_advantages(rewards[:, -1], batch["value_estimates"])

                policy_loss = -(chosen * advantages).mean()
                loss = policy_loss + batch.get("kl_penalty", 0.0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return {"loss": total_loss / max(1, len(rollouts) * self.config.ppo_epochs)}
