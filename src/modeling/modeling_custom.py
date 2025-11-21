"""Custom decoder-only Transformer implementation."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.quantization

from configs.model.base import ModelIOConfig, TransformerConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10_000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :rotary_dim]
    sin = sin[None, None, :, :rotary_dim]
    q_rot = (q[..., :rotary_dim] * cos) + (rotate_half(q[..., :rotary_dim]) * sin)
    k_rot = (k[..., :rotary_dim] * cos) + (rotate_half(k[..., :rotary_dim]) * sin)
    q = torch.cat((q_rot, q[..., rotary_dim:]), dim=-1)
    k = torch.cat((k_rot, k[..., rotary_dim:]), dim=-1)
    return q, k


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.rotary_dim = int(self.head_dim * config.rotary_pct)
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seq_len, hidden_states.device, hidden_states.dtype)
        q, k = apply_rotary_emb(q, k, cos, sin, self.rotary_dim)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(context)


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.silu(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(hidden_states)
        attn_output = self.self_attn(attn_input, attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)
        mlp_input = self.norm2(hidden_states)
        mlp_output = self.mlp(mlp_input)
        return hidden_states + self.dropout(mlp_output)


class CustomTransformer(nn.Module):
    def __init__(self, config: TransformerConfig, io_config: Optional[ModelIOConfig] = None):
        super().__init__()
        self.config = config
        self.io_config = io_config or ModelIOConfig()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.io_config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = config.gradient_checkpointing

    def _build_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = attention_mask.shape
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=attention_mask.device),
            diagonal=1,
        )
        mask = causal_mask[None, None, :, :]  # (1, 1, seq, seq)
        expanded = attention_mask[:, None, None, :].to(dtype=mask.dtype)
        expanded = (1.0 - expanded) * -1e9
        return mask + expanded

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
        hidden_states = self.embed_tokens(input_ids)
        attn_mask = self._build_attention_mask(attention_mask)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint.checkpoint(layer, hidden_states, attn_mask)
            else:
                hidden_states = layer(hidden_states, attn_mask)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def quantize_for_cpu(self, dtype: torch.dtype = torch.qint8) -> "CustomTransformer":
        """Return a dynamically-quantized copy tailored for CPU-only inference."""

        quantized = torch.quantization.quantize_dynamic(
            self, {nn.Linear}, dtype=dtype, inplace=False
        )
        return quantized
