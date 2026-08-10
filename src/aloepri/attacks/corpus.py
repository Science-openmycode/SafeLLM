from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase


def _record_text(record: Mapping[str, Any]) -> str | None:
    for field in ("text", "content", "prompt", "question", "instruction", "output"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_corpus_texts(paths: Iterable[Path]) -> list[str]:
    """Load UTF-8 text, JSON, or JSONL corpora without assuming a dataset provider."""

    texts: list[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".jsonl":
            records: list[Any] = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else [payload]
        else:
            value = path.read_text(encoding="utf-8").strip()
            records = [value] if value else []
        for record in records:
            if isinstance(record, str) and record.strip():
                texts.append(record.strip())
            elif isinstance(record, Mapping):
                text = _record_text(record)
                if text is not None:
                    texts.append(text)
    if not texts:
        raise ValueError("corpus contains no readable text records")
    return texts


def tokenize_corpus(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
) -> list[int]:
    tokens: list[int] = []
    separator = tokenizer.eos_token_id
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            raise TypeError("tokenizer returned batched IDs for one text")
        tokens.extend(int(item) for item in encoded)
        if separator is not None:
            tokens.append(int(separator))
    if not tokens:
        raise ValueError("corpus tokenization produced no token IDs")
    return tokens


def fixed_token_windows(
    token_ids: list[int],
    *,
    sequence_length: int,
    count: int,
    seed: int,
) -> list[list[int]]:
    if sequence_length < 1 or count < 1:
        raise ValueError("sequence_length and count must be positive")
    windows = [
        token_ids[start : start + sequence_length]
        for start in range(0, len(token_ids) - sequence_length + 1, sequence_length)
    ]
    if len(windows) < count:
        raise ValueError(f"need {count} non-overlapping token windows, found {len(windows)}")
    import torch

    order = torch.randperm(len(windows), generator=torch.Generator().manual_seed(seed))[:count]
    return [windows[index] for index in order.tolist()]
