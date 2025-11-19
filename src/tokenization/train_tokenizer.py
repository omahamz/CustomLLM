#!/usr/bin/env python
"""Train a SentencePiece tokenizer on mixed corpora."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterator, List

import sentencepiece as spm

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pq = None  # type: ignore


def iter_text_from_file(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield line
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                text = record.get("text")
                if text:
                    yield str(text)
    elif suffix == ".parquet":
        if pq is None:
            raise SystemExit("pyarrow is required to read parquet files")
        table = pq.read_table(path, columns=["text"])
        for chunk in table.column(0).to_pylist():
            if chunk:
                yield str(chunk)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def iter_text_from_path(path: Path) -> Iterator[str]:
    if path.is_file():
        yield from iter_text_from_file(path)
        return

    for file in sorted(path.rglob("*")):
        if file.suffix.lower() in {".txt", ".jsonl", ".parquet"}:
            yield from iter_text_from_file(file)


def materialize_corpus(paths: List[Path]) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with tmp as handle:
        for source_path in paths:
            for text in iter_text_from_path(source_path):
                handle.write(text.encode("utf-8"))
                handle.write(b"\n")
    return Path(tmp.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Files or directories containing training text.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizer"),
        help="Directory to place SentencePiece artifacts.",
    )
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["unigram", "bpe"],
        default="unigram",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage passed to SentencePiece.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [Path(p).expanduser().resolve() for p in args.input]
    corpus_path = materialize_corpus(sources)

    model_prefix = output_dir / "tokenizer"
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        model_prefix=str(model_prefix),
        split_digits=True,
        byte_fallback=True,
        unk_surface="<unk>",
    )

    metadata = {
        "sources": [str(p) for p in sources],
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
    }
    with (output_dir / "tokenizer.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved tokenizer to {model_prefix}.model")


if __name__ == "__main__":
    main()
