"""Doc-grounded QA evaluation harness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_questions(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
    else:
        raise ValueError("Only JSONL inputs are supported for doc QA")
    return records


def generate_answer(model, tokenizer, question: str, context: str, max_new_tokens: int) -> str:
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=False,
    )
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion[len(prompt) :].strip()


def evaluate(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    total = len(predictions)
    exact = 0
    for pred, refs in zip(predictions, references):
        pred_norm = pred.strip().lower()
        refs_norm = [r.strip().lower() for r in refs]
        if pred_norm in refs_norm:
            exact += 1
    return {"exact_match": exact / max(1, total)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL doc QA dataset")
    parser.add_argument("--max-new", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    records = load_questions(args.dataset)
    predictions: List[str] = []
    references: List[List[str]] = []
    for record in records:
        pred = generate_answer(model, tokenizer, record["question"], record["context"], args.max_new)
        predictions.append(pred)
        references.append(record.get("answers", []))
    metrics = evaluate(predictions, references)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
