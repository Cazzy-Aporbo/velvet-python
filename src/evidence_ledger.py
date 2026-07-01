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

RECOMMENDED_MANIFEST_FIELDS = {
    "run_duration_seconds",
    "split_profile",
}

DATASET_PROFILE_FIELDS = {
    "total_records",
    "label_distribution",
    "label_count",
    "label_imbalance",
}

SPLIT_PROFILE_FIELDS = {
    "train_label_distribution",
    "test_label_distribution",
    "shared_labels",
    "train_only_labels",
    "test_only_labels",
    "train_hash",
    "test_hash",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    """Parse ISO timestamps used in manifests."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO timestamp string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO timestamp") from exc


def _validate_count_map(name: str, payload: Any) -> dict[str, int]:
    """Validate simple label/count mappings."""
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a dictionary")

    normalized: dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} contains non-string key: {key!r}")
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}[{key!r}] must be a non-negative integer")
        normalized[key] = value
    return normalized


def _validate_dataset_profile(payload: dict[str, Any], total_samples: int) -> None:
    """Validate dataset-level quality signals."""
    dataset_profile = payload.get("dataset_profile")
    if not isinstance(dataset_profile, dict):
        raise TypeError("dataset_profile must be a dictionary")

    missing = sorted(DATASET_PROFILE_FIELDS - dataset_profile.keys())
    if missing:
        raise ValueError(f"dataset_profile is missing required fields: {missing}")

    if dataset_profile["total_records"] != total_samples:
        raise ValueError("dataset_profile.total_records must match total_samples")

    label_distribution = _validate_count_map(
        "dataset_profile.label_distribution",
        dataset_profile["label_distribution"],
    )
    if sum(label_distribution.values()) != total_samples:
        raise ValueError("dataset_profile.label_distribution must sum to total_samples")

    label_count = dataset_profile["label_count"]
    if not isinstance(label_count, int) or label_count <= 0:
        raise ValueError("dataset_profile.label_count must be a positive integer")
    if label_count != len(label_distribution):
        raise ValueError("dataset_profile.label_count must match label_distribution cardinality")

    label_imbalance = dataset_profile["label_imbalance"]
    if not isinstance(label_imbalance, float | int):
        raise TypeError("dataset_profile.label_imbalance must be numeric")
    if not 0.0 <= float(label_imbalance) <= 1.0:
        raise ValueError("dataset_profile.label_imbalance must be in [0.0, 1.0]")


def _validate_split_profile(payload: dict[str, Any]) -> None:
    """Validate optional split evidence when present."""
    if "split_profile" not in payload:
        return

    split_profile = payload["split_profile"]
    if not isinstance(split_profile, dict):
        raise TypeError("split_profile must be a dictionary")

    missing = sorted(SPLIT_PROFILE_FIELDS - split_profile.keys())
    if missing:
        raise ValueError(f"split_profile is missing required fields: {missing}")

    train_distribution = _validate_count_map(
        "split_profile.train_label_distribution",
        split_profile["train_label_distribution"],
    )
    test_distribution = _validate_count_map(
        "split_profile.test_label_distribution",
        split_profile["test_label_distribution"],
    )
    if sum(train_distribution.values()) != payload["train_size"]:
        raise ValueError("split_profile.train_label_distribution must sum to train_size")
    if sum(test_distribution.values()) != payload["test_size"]:
        raise ValueError("split_profile.test_label_distribution must sum to test_size")

    for field in ("shared_labels", "train_only_labels", "test_only_labels"):
        values = split_profile[field]
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise TypeError(f"split_profile.{field} must be a list of strings")

    shared = sorted(set(split_profile["shared_labels"]))
    train_only = sorted(set(split_profile["train_only_labels"]))
    test_only = sorted(set(split_profile["test_only_labels"]))
    train_labels = set(train_distribution)
    test_labels = set(test_distribution)

    if shared != sorted(train_labels & test_labels):
        raise ValueError("split_profile.shared_labels does not match label intersection")
    if train_only != sorted(train_labels - test_labels):
        raise ValueError("split_profile.train_only_labels does not match split reality")
    if test_only != sorted(test_labels - train_labels):
        raise ValueError("split_profile.test_only_labels does not match split reality")

    for field in ("train_hash", "test_hash"):
        value = split_profile[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"split_profile.{field} must be a non-empty string")


def _validate_label_contract(payload: dict[str, Any]) -> None:
    """Validate label-oriented structures inside a manifest."""
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(item, str) for item in labels):
        raise ValueError("labels must be a non-empty list of strings")

    class_coverage = _validate_count_map("class_coverage", payload.get("class_coverage"))
    if sorted(class_coverage) != sorted(labels):
        raise ValueError("class_coverage keys must match labels")

    confusion_matrix = payload.get("confusion_matrix")
    if not isinstance(confusion_matrix, dict):
        raise TypeError("confusion_matrix must be a dictionary of dictionaries")
    for actual_label, predicted_counts in confusion_matrix.items():
        if not isinstance(actual_label, str):
            raise TypeError("confusion_matrix contains non-string row labels")
        _validate_count_map(f"confusion_matrix[{actual_label!r}]", predicted_counts)


def _recommendations_from_alerts(alerts: list[dict[str, Any]]) -> list[str]:
    """Translate drift alerts into practical next actions."""
    kinds = {alert["kind"] for alert in alerts}
    recommendations: list[str] = []

    if "dataset_hash_drift" in kinds:
        recommendations.append(
            "Pin the dataset source before comparing runs so the evidence trail stays meaningful."
        )
    if "test_ratio_drift" in kinds:
        recommendations.append(
            "Keep the holdout ratio fixed while benchmarking models; change one variable at a time."
        )
    if "accuracy_spread" in kinds:
        recommendations.append(
            "When seed variance widens, review class balance and compare confusion matrices instead of headline accuracy alone."
        )
    if "model_dataset_inconsistency" in kinds:
        recommendations.append(
            "A model family used different dataset snapshots; rerun after freezing ingestion and artifact paths."
        )
    if not recommendations:
        recommendations.append(
            "The ledger is stable; the next useful step is to rerun on another seed range and compare the two ledgers."
        )

    return recommendations


def _health_summary(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact health view for reviewers."""
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for alert in alerts:
        severity = alert.get("severity", "low")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    if severity_counts["high"]:
        status = "action_required"
    elif severity_counts["medium"]:
        status = "review_recommended"
    else:
        status = "stable"

    return {
        "status": status,
        "alert_count": len(alerts),
        "severity_counts": severity_counts,
    }


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

    train_size = payload["train_size"]
    test_size = payload["test_size"]
    if not isinstance(train_size, int) or train_size <= 0:
        raise ValueError("train_size must be a positive integer")
    if not isinstance(test_size, int) or test_size <= 0:
        raise ValueError("test_size must be a positive integer")
    if train_size + test_size != sample_count:
        raise ValueError("train_size and test_size must sum to total_samples")

    started_at = _parse_utc_timestamp(payload["started_at"], field="started_at")
    finished_at = _parse_utc_timestamp(payload["finished_at"], field="finished_at")
    if finished_at < started_at:
        raise ValueError("finished_at must be greater than or equal to started_at")

    if "run_duration_seconds" in payload:
        duration = payload["run_duration_seconds"]
        if not isinstance(duration, float | int):
            raise TypeError("run_duration_seconds must be numeric")
        if float(duration) < 0:
            raise ValueError("run_duration_seconds must be non-negative")

    _validate_dataset_profile(payload, sample_count)
    _validate_split_profile(payload)
    _validate_label_contract(payload)

    missing_recommended = sorted(RECOMMENDED_MANIFEST_FIELDS - payload.keys())
    if missing_recommended:
        payload = dict(payload)
        payload["recommended_missing_fields"] = missing_recommended

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
    alerts = drift_checks(grouped, threshold=accuracy_spread_threshold)

    return {
        "generated_at": _utc_now(),
        "ledger_schema": LEDGER_SCHEMA_VERSION,
        "run_count": len(manifests),
        "model_count": len(grouped),
        "runs": [manifest["run_id"] for manifest in manifests],
        "model_summaries": summaries,
        "drift_alerts": alerts,
        "health": _health_summary(alerts),
        "recommendations": _recommendations_from_alerts(alerts),
    }


def write_evidence_ledger(ledger: dict[str, Any], path: str | Path) -> Path:
    """Write ledger JSON to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return output
