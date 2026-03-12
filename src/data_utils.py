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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml



def load_dataset() -> List[Tuple[str, str]]:
    """Return a small labeled corpus for testing classifiers.

    The data is chosen so that:
        - "greeting" class has overlapping vocabulary ("hello")
        - "tech" class shares "learning" across documents
        - "unknown" label exists for out-of-vocabulary inputs
        - Each class has at least two examples for meaningful training

    This makes the dataset useful for verifying train/predict/evaluate
    without external files or network access.
    """
    return [
        ("hello world", "greeting"),
        ("hello there friend", "greeting"),
        ("hi hello good morning", "greeting"),
        ("machine learning algorithms", "tech"),
        ("deep learning neural networks", "tech"),
        ("natural language processing", "tech"),
        ("the weather is nice today", "casual"),
        ("going to the park later", "casual"),
        ("unseen phrase with no match", "unknown"),
    ]


def train_test_split(
    data: List[Tuple[str, str]],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split labeled data into training and test sets.

    Uses a deterministic shuffle (seeded) so results are reproducible.
    The split point is floor(len(data) * (1 - test_ratio)).

    This avoids pulling in sklearn just for train_test_split — which
    is one of the most over-imported functions in data science.
    """
    import random

    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]



def load_csv(
    path: Union[str, Path],
    text_col: str = "text",
    label_col: str = "label",
    delimiter: str = ",",
) -> List[Tuple[str, str]]:
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

        return [
            (row[text_col].strip(), row[label_col].strip())
            for row in reader
            if row[text_col].strip() and row[label_col].strip()
        ]



def load_config(config_path: Union[str, Path]) -> dict:
    """Parse a YAML file and return its contents as a dictionary.

    YAML is used here instead of JSON because it supports comments,
    which makes configuration files self-documenting. The trade-off
    is a runtime dependency on PyYAML.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}