"""
Data loading, transformation, and validation utilities.

Three loading strategies are provided:
    1. load_dataset()      — hardcoded corpus for unit tests
    2. load_csv()          — parse any CSV with header row
    3. load_config()       — YAML configuration files

The hardcoded dataset is designed so that each classifier in
src/ai.py and src/ml_pipeline.py can be tested deterministically
without touching the filesystem.

For real data, load_csv handles the common case: read a file,
pick two columns, return (text, label) pairs. It validates
that the requested columns exist and raises clear errors if not.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from hashlib import sha256
from pathlib import Path

import yaml

_SUPPORTED_TEXT_TYPES = (str,)


def validate_dataset(data: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Validate and normalize labeled text records.

    The function intentionally fails fast on malformed input so learning
    pipelines do not silently train on bad rows.

    Validation rules:
        - each item is a 2-tuple of (text, label)
        - both fields are non-empty strings
        - leading/trailing whitespace is removed
    """

    normalized: list[tuple[str, str]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(f"Row {idx} is not a 2-tuple: {row!r}")

        text, label = row
        if not isinstance(text, _SUPPORTED_TEXT_TYPES):
            raise TypeError(f"Row {idx} text is not a string: {type(text)!r}")
        if not isinstance(label, _SUPPORTED_TEXT_TYPES):
            raise TypeError(f"Row {idx} label is not a string: {type(label)!r}")

        text_clean = text.strip()
        label_clean = label.strip()
        if not text_clean:
            raise ValueError(f"Row {idx} has empty text")
        if not label_clean:
            raise ValueError(f"Row {idx} has empty label")

        normalized.append((text_clean, label_clean))

    if not normalized:
        raise ValueError("No usable rows after validation")

    return normalized


def dataset_signature(data: Iterable[tuple[str, str]]) -> str:
    """Create a stable cryptographic signature for dataset provenance.

    The signature is invariant to row order by sorting normalized rows first.
    This allows different callers to compare dataset integrity even when
    input sequences are produced from different loaders or iteration paths.
    """
    normalized = validate_dataset(data)
    ordered = sorted(normalized)
    payload = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def dataset_profile(data: Iterable[tuple[str, str]]) -> dict[str, object]:
    """Return a compact profile for a labeled text dataset.

    The profile is intentionally lightweight and dependency-free:
    - total records and label cardinality,
    - duplicate row count,
    - duplicate text count,
    - min/mean/max record length,
    - simple label concentration score.

    This is meant for engineering checkpoints and CI visibility.
    """
    normalized = validate_dataset(data)

    texts = [text for text, _ in normalized]
    labels = [label for _, label in normalized]
    text_counter = Counter(texts)
    lengths = [len(text) for text in texts]

    # Shannon-style concentration across classes; ranges from 0..1.
    total = len(labels)
    probs = [count / total for count in Counter(labels).values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(Counter(labels))) if len(Counter(labels)) > 1 else 0.0
    concentration = 1.0 - (entropy / max_entropy if max_entropy else 0.0)

    return {
        "total_records": len(normalized),
        "unique_records": len(set(normalized)),
        "duplicate_records": len(normalized) - len(set(normalized)),
        "unique_texts": len(text_counter),
        "duplicated_texts": sum(1 for count in text_counter.values() if count > 1),
        "min_text_length": min(lengths),
        "max_text_length": max(lengths),
        "avg_text_length": round(sum(lengths) / len(lengths), 3),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "label_count": len(set(labels)),
        "label_imbalance": round(concentration, 4),
    }


def label_distribution(data: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Return class frequency map from labeled dataset."""
    return Counter(label for _, label in data)


def load_dataset() -> list[tuple[str, str]]:
    """Return a small labeled corpus for testing classifiers.

    The data is chosen so that:
        - "greeting" class has overlapping vocabulary ("hello")
        - "tech" class shares "learning" across documents
        - "unknown" label exists for out-of-vocabulary inputs
        - Each class has at least two examples for meaningful training

    This makes the dataset useful for verifying train/predict/evaluate
    without external files or network access.
    """
    return validate_dataset([
        ("hello world", "greeting"),
        ("hello there friend", "greeting"),
        ("hi hello good morning", "greeting"),
        ("machine learning algorithms", "tech"),
        ("deep learning neural networks", "tech"),
        ("natural language processing", "tech"),
        ("the weather is nice today", "casual"),
        ("going to the park later", "casual"),
        ("unseen phrase with no match", "unknown"),
    ])


def train_test_split(
    data: list[tuple[str, str]],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split labeled data into training and test sets.

    Uses a deterministic shuffle (seeded) so results are reproducible.
    The split point is floor(len(data) * (1 - test_ratio)).

    This avoids pulling in sklearn just for train_test_split — which
    is one of the most over-imported functions in data science.
    """
    import random

    data = validate_dataset(data)
    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def iter_batches(
    data: Sequence[tuple[str, str]],
    batch_size: int = 32,
) -> Iterator[list[tuple[str, str]]]:
    """Yield fixed-size batches for lightweight stream-like processing."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(data), batch_size):
        yield data[start : start + batch_size]



def load_csv(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    delimiter: str = ",",
) -> list[tuple[str, str]]:
    """Load (text, label) pairs from a CSV file.

    Validates that both columns exist in the header row.
    Strips whitespace from values. Skips rows where either
    column is empty.

    Raises:
        FileNotFoundError: if the path does not exist.
        KeyError: if text_col or label_col is missing from the header.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file at {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise KeyError("CSV file has no header row")

        missing = {text_col, label_col} - set(reader.fieldnames)
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        return validate_dataset([
            (row[text_col].strip(), row[label_col].strip())
            for row in reader
            if row[text_col].strip() and row[label_col].strip()
        ])



def load_config(config_path: str | Path) -> dict:
    """Parse a YAML file and return its contents as a dictionary.

    YAML is used here instead of JSON because it supports comments,
    which makes configuration files self-documenting. The trade-off
    is a runtime dependency on PyYAML.
    """
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping (top-level dict), got {type(config).__name__}"
        )
    return config
