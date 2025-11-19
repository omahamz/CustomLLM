#!/usr/bin/env python
"""Stream HuggingFace datasets, clean documents, and shard them to Parquet."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import datasets as hf_datasets

try:  # pragma: no cover - import path depends on installed version
    from datasets.exceptions import DatasetNotFoundError
except ImportError:  # pragma: no cover - compatibility shim
    try:
        from datasets import DatasetNotFoundError  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - extremely old versions
        DatasetNotFoundError = RuntimeError
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------------------------------
CODE_BLOCK_RE = re.compile(r"```(?:(?:[a-zA-Z0-9_+-]+)\n)?(.*?)```", re.DOTALL)
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NL_RE = re.compile(r"\n{3,}")


def extract_code_blocks(text: str) -> List[str]:
    """Return all fenced code blocks while keeping their inner content."""
    return [match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text)]


def clean_code_blocks(text: str) -> str:
    """Replace fenced code blocks with inline XML-like tags.

    Wrapping code with ``<code>`` tags makes downstream parsing easier and
    prevents us from dropping valuable indentation information.
    """

    def _replace(match: re.Match[str]) -> str:
        block = match.group(1).strip("\n")
        return f"\n<code>\n{block}\n</code>\n"

    return CODE_BLOCK_RE.sub(_replace, text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Dataset streaming and sharding
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    name: str
    split: str
    text_field: str

    @classmethod
    def parse(cls, spec: str) -> "DatasetSpec":
        """Parse strings of the form ``dataset:split:text``."""
        try:
            dataset, split, text_field = spec.split(":", maxsplit=2)
        except ValueError as exc:  # pragma: no cover - defensive
            raise SystemExit(
                "Dataset specs must look like 'name:split:text_field'"
            ) from exc
        return cls(dataset, split, text_field)


def stream_dataset(spec: DatasetSpec) -> Iterator[Dict]:
    """Yield examples from a Hugging Face dataset split.

    Some datasets (e.g., ``bigcode/the-stack-dedup``) are gated on the Hub and
    require authentication.  Surfacing that requirement early is more helpful
    than the default ``datasets`` exception, so we catch it and replace it with
    an actionable message.
    """

    try:
        ds = hf_datasets.load_dataset(spec.name, split=spec.split, streaming=True)
    except DatasetNotFoundError as exc:  # pragma: no cover - network side-effect
        raise SystemExit(
            "Failed to load dataset"
            f" '{spec.name}:{spec.split}'. This dataset may be gated on the"
            " Hugging Face Hub. Run `huggingface-cli login` or set the"
            " `HF_TOKEN` environment variable before re-running the script."
        ) from exc

    for example in ds:
        yield example


def process_example(example: Dict, text_field: str) -> Dict:
    if text_field not in example:
        raise KeyError(f"Missing field '{text_field}' in example: {example.keys()}")

    raw_text = str(example[text_field])
    cleaned = clean_code_blocks(raw_text)
    normalized = normalize_whitespace(cleaned)
    code_blocks = extract_code_blocks(raw_text)

    processed = {
        "text": normalized,
        "num_chars": len(normalized),
        "num_code_blocks": len(code_blocks),
    }
    if code_blocks:
        processed["code_blocks"] = "\n\n".join(code_blocks)
    if "license" in example:
        processed["license"] = example["license"]
    return processed


def write_shard(records: List[Dict], output_dir: Path, shard_idx: int) -> Path:
    table = pa.Table.from_pylist(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard-{shard_idx:05d}.parquet"
    pq.write_table(table, shard_path)
    return shard_path


def shard_stream(
    spec: DatasetSpec,
    output_root: Path,
    shard_size: int,
) -> None:
    buffer: List[Dict] = []
    shard_idx = 0
    dataset_dir = output_root / spec.name.replace("/", "_") / spec.split
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for example in stream_dataset(spec):
        processed = process_example(example, spec.text_field)
        buffer.append(processed)
        if len(buffer) >= shard_size:
            shard_idx += 1
            write_shard(buffer, dataset_dir, shard_idx)
            buffer.clear()

    if buffer:
        shard_idx += 1
        write_shard(buffer, dataset_dir, shard_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec formatted as 'name:split:text_field'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public/processed"),
        help="Directory to place sharded Parquet files.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2_000,
        help="Number of records per Parquet shard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir

    for raw_spec in args.dataset:
        spec = DatasetSpec.parse(raw_spec)
        print(f"Processing {spec.name}:{spec.split} -> {output_root}")
        shard_stream(spec, output_root, args.shard_size)

    print("Done.")


if __name__ == "__main__":
    main()
