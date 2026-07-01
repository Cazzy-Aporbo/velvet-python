"""Ledger utilities for reproducible experiment evidence.

This module is the credibility layer for the repository:
- validate manifest contracts
- aggregate model results
- expose deterministic drift signals for reviews and incident analysis
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline import MANIFEST_SCHEMA_VERSION, ExperimentRun

LEDGER_SCHEMA_VERSION = "vp-evidence-ledger-v1"

MIN_REQUIRED_MANIFEST_FIELDS = {
    "run_id",
    "model_name",
    "model_type",
    "seed",
    "test_ratio",
    "train_size",
    "test_size",
    "total_samples",
    "accuracy",
    "dataset_hash",
    "dataset_profile",
    "started_at",
    "finished_at",
    "python_version",
    "manifest_schema",
    "parameters",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_run_dict(run: ExperimentRun | dict[str, Any]) -> dict[str, Any]:
    """Convert a run artifact to a dict for aggregation."""
    if isinstance(run, dict):
        return dict(run)
    if hasattr(run, "__dataclass_fields__"):
        return asdict(run)
    raise TypeError(f"Unsupported run payload type: {type(run)!r}")


def validate_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a single manifest contract.

    The validation here is intentionally strict for educational reliability:
    if evidence is required, schema drift should fail loudly.
    """
    missing = sorted(MIN_REQUIRED_MANIFEST_FIELDS - payload.keys())
    if missing:
        raise ValueError(f"manifest is missing required fields: {missing}")

    accuracy = payload.get("accuracy")
    if not isinstance(accuracy, float | int):
        raise TypeError("accuracy must be numeric")
    if not 0.0 <= float(accuracy) <= 1.0:
        raise ValueError("accuracy must be in [0.0, 1.0]")

    schema = payload["manifest_schema"]
    if not str(schema).startswith("vp-manifest-"):
        raise ValueError(f"unexpected manifest schema: {schema!r}")

    sample_count = payload["total_samples"]
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("total_samples must be a positive integer")

    return payload


def summarize_model_group(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary + variance metrics for one model bucket."""
    seeds = sorted({entry["seed"] for entry in entries})
    accuracies = sorted(float(entry["accuracy"]) for entry in entries)
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    mean_accuracy = sum(accuracies) / len(accuracies)
    variance = sum((v - mean_accuracy) ** 2 for v in accuracies) / len(accuracies)
    std_accuracy = math.sqrt(variance)

    dataset_hashes = sorted({entry["dataset_hash"] for entry in entries})
    test_ratios = sorted({entry["test_ratio"] for entry in entries})
    profile = entries[0]["dataset_profile"]

    model_name = entries[0]["model_name"]
    model_type = entries[0]["model_type"]
    return {
        "model_name": model_name,
        "model_type": model_type,
        "run_count": len(entries),
        "seeds": seeds,
        "accuracy": {
            "min": round(min_accuracy, 6),
            "max": round(max_accuracy, 6),
            "mean": round(mean_accuracy, 6),
            "spread": round(max_accuracy - min_accuracy, 6),
            "std": round(std_accuracy, 6),
        },
        "dataset_hashes": dataset_hashes,
        "dataset_hash_stable": len(dataset_hashes) == 1,
        "test_ratio_values": test_ratios,
        "label_coverage": profile.get("label_distribution", {}),
    }


def drift_checks(groups: dict[str, list[dict[str, Any]]], threshold: float = 0.05) -> list[dict[str, Any]]:
    """Build a deterministic list of meaningful signal changes across runs."""
    alerts: list[dict[str, Any]] = []

    all_hashes = sorted({entry["dataset_hash"] for runs in groups.values() for entry in runs})
    if len(all_hashes) > 1:
        alerts.append(
            {
                "kind": "dataset_hash_drift",
                "severity": "high",
                "message": "manifest dataset hashes differ across runs",
                "dataset_hashes": all_hashes,
            },
        )

    all_test_ratios = sorted(
        {entry["test_ratio"] for runs in groups.values() for entry in runs}
    )
    if len(all_test_ratios) > 1:
        alerts.append(
            {
                "kind": "test_ratio_drift",
                "severity": "medium",
                "message": "test ratios differ across runs",
                "test_ratio_values": all_test_ratios,
            },
        )

    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        summary = summarize_model_group(entries)
        spread = summary["accuracy"]["spread"]
        if spread > threshold:
            alerts.append(
                {
                    "kind": "accuracy_spread",
                    "severity": "medium",
                    "model": key,
                    "spread": spread,
                    "message": (
                        "accuracy spread exceeded threshold within same model family; "
                        "compare seeds and split randomness."
                    ),
                },
            )
        if not summary["dataset_hash_stable"]:
            alerts.append(
                {
                    "kind": "model_dataset_inconsistency",
                    "severity": "high",
                    "model": key,
                    "message": (
                        "same model + seed family used different dataset hashes; "
                        "check input ingestion path."
                    ),
                },
            )

    return alerts


def build_evidence_ledger(
    runs: Iterable[ExperimentRun | dict[str, Any]],
    *,
    accuracy_spread_threshold: float = 0.05,
) -> dict[str, Any]:
    """Build a deterministic evidence ledger from experiment runs."""
    manifests = [validate_manifest_payload(_to_run_dict(run)) for run in runs]
    if not manifests:
        raise ValueError("No runs provided for evidence ledger")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in sorted(manifests, key=lambda payload: (payload["model_name"], payload["seed"])):
        if not manifest["manifest_schema"].startswith("vp-manifest-"):
            raise ValueError("Unsupported manifest schema; expected vp-manifest-*")

        if manifest["manifest_schema"] != MANIFEST_SCHEMA_VERSION:
            # Strict lock-step contract to avoid invisible compatibility breaks.
            raise ValueError(
                f"Unsupported manifest schema version: {manifest['manifest_schema']}"
            )

        key = f"{manifest['model_name']}::{manifest['model_type']}"
        grouped[key].append(manifest)

    summaries = [
        summarize_model_group(entries)
        for _, entries in sorted(grouped.items(), key=lambda item: item[0])
    ]

    return {
        "generated_at": _utc_now(),
        "ledger_schema": LEDGER_SCHEMA_VERSION,
        "run_count": len(manifests),
        "model_count": len(grouped),
        "runs": [manifest["run_id"] for manifest in manifests],
        "model_summaries": summaries,
        "drift_alerts": drift_checks(grouped, threshold=accuracy_spread_threshold),
    }


def write_evidence_ledger(ledger: dict[str, Any], path: str | Path) -> Path:
    """Write ledger JSON to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return output
