"""Utilities for monitoring and adapting to system memory limits."""
from __future__ import annotations

import psutil


def available_ram_bytes() -> int:
    return int(psutil.virtual_memory().available)


def format_gibibytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.1f} GiB"


def suggest_batch_size(
    requested_batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    safety_margin: float = 0.6,
) -> int:
    """Estimate a conservative batch size based on available system RAM.

    The estimation multiplies per-token activation memory by sequence length and
    number of layers, then applies a safety margin to reduce the risk of system
    swapping.
    """

    available = int(available_ram_bytes() * safety_margin)
    # Rough activation budget: embeddings + attention + MLP (~12x hidden_size)
    bytes_per_token = hidden_size * 12 * 4  # float32 baseline
    bytes_per_sample = max(seq_length * bytes_per_token * (1 + num_layers / 24), 1)
    max_batch = max(1, available // bytes_per_sample)
    return max(1, min(requested_batch_size, int(max_batch)))


def memory_snapshot() -> str:
    vm = psutil.virtual_memory()
    used = vm.total - vm.available
    return f"RAM {format_gibibytes(used)} / {format_gibibytes(vm.total)} ({vm.percent:.1f}% used)"


def near_oom(threshold_pct: float = 92.0) -> bool:
    """Return True when system memory usage exceeds the given percentage."""

    return psutil.virtual_memory().percent >= threshold_pct
