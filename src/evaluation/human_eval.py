"""HumanEval and MBPP evaluation runners."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import datasets
import torch
from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoModelForCausalLM, AutoTokenizer


def sample_code(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completion[len(prompt) :]


def run_humaneval(model_name: str, max_new_tokens: int) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = datasets.load_dataset("openai_humaneval", split="test")
    samples: List[Dict[str, str]] = []
    problems: List[Dict[str, str]] = []
    for record in dataset:
        prompt = record["prompt"]
        completion = sample_code(model, tokenizer, prompt, max_new_tokens)
        samples.append({"task_id": record["task_id"], "completion": completion})
        problems.append(
            {
                "task_id": record["task_id"],
                "prompt": record["prompt"],
                "canonical_solution": record["canonical_solution"],
                "test": record["test"],
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        samples_path = tmp_path / "samples.jsonl"
        problems_path = tmp_path / "problems.jsonl"
        write_jsonl(str(samples_path), samples)
        write_jsonl(str(problems_path), problems)
        result = evaluate_functional_correctness(
            sample_file=str(samples_path),
            problem_file=str(problems_path),
            k=[1, 5],
        )
    return result


def run_mbpp(model_name: str, max_new_tokens: int) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = datasets.load_dataset("mbpp", split="test")
    correct = 0
    for record in dataset:
        prompt = record["text"]
        completion = sample_code(model, tokenizer, prompt, max_new_tokens)
        candidate = record.get("code", "") + "\n" + completion
        tests = record.get("test_list", [])
        if isinstance(tests, str):
            try:
                tests = json.loads(tests)
            except json.JSONDecodeError:
                tests = [tests]
        if execute_tests(candidate, tests):
            correct += 1
    return {"accuracy": correct / len(dataset)}


def execute_tests(code: str, tests: List[str]) -> bool:
    namespace: Dict[str, object] = {}
    try:
        exec(code, namespace)
        for test in tests:
            if not eval(test, namespace):
                return False
    except Exception:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model identifier")
    parser.add_argument(
        "--benchmark",
        choices=["humaneval", "mbpp"],
        default="humaneval",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.benchmark == "humaneval":
        result = run_humaneval(args.model, args.max_new_tokens)
    else:
        result = run_mbpp(args.model, args.max_new_tokens)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
