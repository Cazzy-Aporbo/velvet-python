"""
Data utilities for the mock ML pipeline.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import List, Tuple


def load_dataset() -> List[Tuple[str, str]]:
    """
    Load a small hardcoded dataset.

    Each sample is (text, label).
    This mimics loading from disk but is self-contained.
    """
    dataset = [
        ("hello ai", "hello"),
        ("hello world", "hello"),
        ("machine learning", "machine"),
        ("unseen phrase", "unknown"),
    ]
    return dataset


def load_config(config_path: str | Path) -> dict:
    """
    Load YAML configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)