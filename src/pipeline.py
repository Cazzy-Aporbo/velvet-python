"""Reproducible experimentation utilities for Velvet Python.

This module lets you run a transparent text classification pipeline with
explicit run metadata and deterministic splits.

The design is intentionally practical:
- validate inputs up front
- preserve deterministic behavior for education and reproducibility
- expose results in a machine-readable manifest
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_utils import (
    dataset_profile,
    dataset_signature,
    label_distribution,
    train_test_split,
    validate_dataset,
)

ModelBuilder = Callable[[], Any]

MANIFEST_SCHEMA_VERSION = "vp-manifest-v2.1"


@dataclass(frozen=True)
class ExperimentRun:
    """Represents one deterministic training/evaluation run."""

    run_id: str
    started_at: str
    finished_at: str
    model_name: str
    model_type: str
    seed: int
    test_ratio: float
    python_version: str
    manifest_schema: str
    train_size: int
    test_size: int
    total_samples: int
    accuracy: float
    class_coverage: dict[str, int]
    labels: list[str]
    confusion_matrix: dict[str, dict[str, int]]
    dataset_profile: dict[str, object]
    dataset_hash: str
    parameters: dict[str, Any]


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_model(
    model: Any,
    train_data: list[tuple[str, str]],
    test_data: list[tuple[str, str]],
) -> tuple[float, dict[str, dict[str, int]]]:
    """Train and evaluate a single model instance."""
    if not hasattr(model, "train") or not hasattr(model, "predict"):
        raise TypeError("Model must expose both train and predict methods.")

    train_texts = [text for text, _ in train_data]
    train_labels = [label for _, label in train_data]

    model.train(train_texts, train_labels)

    test_texts = [text for text, _ in test_data]
    test_labels = [label for _, label in test_data]

    predictions: list[str] = [model.predict(text) for text in test_texts]
    correct = sum(pred == label for pred, label in zip(predictions, test_labels, strict=False))
    accuracy = 0.0 if not test_texts else correct / len(test_texts)

    matrix: dict[str, dict[str, int]] = {}
    all_labels = sorted(set(test_labels).union(predictions))
    for actual in all_labels:
        matrix[actual] = dict.fromkeys(all_labels, 0)

    for pred, (_, actual) in zip(predictions, test_data, strict=False):
        matrix.setdefault(actual, {})
        matrix[actual][pred] = matrix[actual].get(pred, 0) + 1

    return accuracy, matrix


def _to_json_dict(item: ExperimentRun) -> dict[str, Any]:
    """Convert run metadata to JSON serializable dict."""
    return asdict(item)


def run_classification_pipeline(
    model_name: str,
    model_builder: ModelBuilder,
    data: Iterable[tuple[str, str]],
    *,
    seed: int = 42,
    test_ratio: float = 0.30,
    parameters: dict[str, Any] | None = None,
) -> ExperimentRun:
    """Execute a full experiment and return reproducible run evidence.

    Parameters
    ----------
    model_name:
        Human-readable model label for the output manifest.
    model_builder:
        A callable that returns an initialized model instance.
    data:
        Iterable of (text, label) tuples.
    seed:
        Random seed for deterministic split.
    test_ratio:
        Proportion of records held out for validation.
    parameters:
        Arbitrary run metadata, such as dataset version or preprocessing settings.
    """

    data_list = validate_dataset(data)
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    train_data, test_data = train_test_split(data_list, test_ratio=test_ratio, seed=seed)
    started_at = _now_utc()

    model = model_builder()
    accuracy, matrix = _run_model(model, train_data, test_data)

    counts = Counter(label for _, label in data_list)
    labels = sorted(counts.keys())

    completed_at = _now_utc()
    safe_model = re.sub(r"[^a-z0-9]+", "-", model_name.lower().strip())
    safe_model = safe_model.strip("-") or "model"

    run = ExperimentRun(
        run_id=f"vp-s{seed}-t{int(test_ratio * 100)}-{safe_model}-{len(data_list)}n",
        started_at=started_at,
        finished_at=completed_at,
        model_name=model_name,
        model_type=type(model).__name__,
        seed=seed,
        test_ratio=test_ratio,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        manifest_schema=MANIFEST_SCHEMA_VERSION,
        train_size=len(train_data),
        test_size=len(test_data),
        total_samples=len(data_list),
        accuracy=round(float(accuracy), 6),
        class_coverage=dict(sorted(label_distribution(data_list).items())),
        labels=labels,
        confusion_matrix=matrix,
        dataset_profile=dataset_profile(data_list),
        dataset_hash=dataset_signature(data_list),
        parameters=parameters or {},
    )

    return run


def run_epochs(
    model_name: str,
    model_builder: ModelBuilder,
    data: Iterable[tuple[str, str]],
    *,
    epochs: int = 3,
    seed: int = 42,
    test_ratio: float = 0.30,
    parameters: dict[str, Any] | None = None,
) -> list[ExperimentRun]:
    """Run deterministic repeated passes with controlled seed offsets.

    This is a learning-oriented scaffold for repeated experiment sweeps.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    records = []
    for i in range(epochs):
        run = run_classification_pipeline(
            model_name=f"{model_name} (epoch={i + 1}/{epochs})",
            model_builder=model_builder,
            data=data,
            seed=seed + i,
            test_ratio=test_ratio,
            parameters={**(parameters or {}), "epoch": i + 1},
        )
        records.append(run)
    return records


def dump_run_manifest(run: ExperimentRun, path: str | Path) -> Path:
    """Persist a run manifest to disk and return the written location."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_dict(run), indent=2), encoding="utf-8")
    return path


def dump_run_manifests(runs: Iterable[ExperimentRun], output_dir: str | Path) -> list[Path]:
    """Persist multiple run manifests and return written file paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in runs:
        filename = f"{run.run_id.replace(' ', '_')}.json"
        paths.append(dump_run_manifest(run, output / filename))
    return paths


def summarize_runs(runs: Iterable[ExperimentRun]) -> list[dict[str, Any]]:
    """Compact performance table used by docs and terminal reports."""
    return [
        {
            "model_name": run.model_name,
            "seed": run.seed,
            "accuracy": run.accuracy,
            "train_size": run.train_size,
            "test_size": run.test_size,
        }
        for run in runs
    ]
