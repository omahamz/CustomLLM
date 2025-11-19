"""Dataclasses describing the Custom Transformer configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 65_536
    hidden_size: int = 3_072
    intermediate_size: int = 8_192
    num_attention_heads: int = 16
    num_layers: int = 24
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8_192
    rotary_pct: float = 0.25
    rope_theta: float = 10_000.0
    gradient_checkpointing: bool = False


@dataclass
class ModelIOConfig:
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_embeddings: bool = True
