"""Compute perplexity over a corpus."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_texts(path: Path) -> Iterable[str]:
    if path.is_dir():
        for file in sorted(path.rglob("*.parquet")):
            yield from iter_texts(file)
        return
    if path.suffix == ".parquet":
        table = pq.read_table(path, columns=["text"])
        for item in table.column(0).to_pylist():
            if item:
                yield str(item)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield line.strip()


def compute_perplexity(model_name: str, corpus_path: Path, max_samples: int, max_length: int) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total_loss = 0.0
    total_tokens = 0
    for idx, text in enumerate(iter_texts(corpus_path)):
        if idx >= max_samples:
            break
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model(**encoded, labels=encoded["input_ids"])
        total_loss += outputs.loss.item() * encoded["input_ids"].size(1)
        total_tokens += encoded["input_ids"].size(1)
    return math.exp(total_loss / max(1, total_tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, help="Optional JSON file to write metrics")
    args = parser.parse_args()

    ppl = compute_perplexity(args.model, args.corpus, args.max_samples, args.max_length)
    metrics = {"perplexity": ppl}
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
