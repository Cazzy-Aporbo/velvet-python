from __future__ import annotations

import pytest

import json
from src.data_utils import dataset_signature, load_dataset
from src.evidence_ledger import (
    LEDGER_SCHEMA_VERSION,
    build_evidence_ledger,
    summarize_model_group,
    validate_manifest_payload,
    write_evidence_ledger,
)
from src.ml_pipeline import WordFrequencyModel
from src.pipeline import run_classification_pipeline


def test_validate_manifest_payload_enforces_contract():
    dataset = load_dataset()
    run = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=7,
    )
    payload = run.__dict__.copy()
    validated = validate_manifest_payload(payload)
    assert validated["run_id"] == run.run_id

    missing_field = payload.copy()
    missing_field.pop("dataset_hash")
    try:
        validate_manifest_payload(missing_field)
    except ValueError as exc:
        assert "missing required fields" in str(exc)
        assert "dataset_hash" in str(exc)


def test_evidence_ledger_raises_on_schema_or_payload_drift():
    dataset = load_dataset()
    run = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=3,
    )
    payload = run.__dict__.copy()
    payload["manifest_schema"] = "bad-schema"

    with pytest.raises(ValueError, match="unexpected manifest schema"):
        validate_manifest_payload(payload)

    payload = run.__dict__.copy()
    payload["manifest_schema"] = "vp-manifest-v2.1"
    payload["accuracy"] = 1.2
    with pytest.raises(ValueError, match="accuracy must be in"):
        validate_manifest_payload(payload)


def test_evidence_ledger_builds_drift_alerts_and_ordering():
    dataset = load_dataset()
    run_a = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=11,
    )
    run_b = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=12,
    )

    # Force a controlled spread to exercise model-level drift detection.
    b_payload = run_b.__dict__.copy()
    b_payload["accuracy"] = (
        run_a.accuracy - 0.06
        if run_a.accuracy >= 0.06
        else run_a.accuracy + 0.06
    )

    ledger = build_evidence_ledger([run_a.__dict__, b_payload], accuracy_spread_threshold=0.05)
    assert ledger["run_count"] == 2
    assert ledger["model_count"] == 1
    assert any(alert["kind"] == "accuracy_spread" for alert in ledger["drift_alerts"])
    assert ledger["model_summaries"][0]["run_count"] == 2

    # stable deterministic ordering for reproducible reviews
    assert ledger["runs"] == sorted(ledger["runs"])


def test_evidence_ledger_detects_dataset_and_split_drift():
    dataset = load_dataset()
    run_a = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=5,
        test_ratio=0.25,
    )
    run_b = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset[:4],
        seed=5,
        test_ratio=0.40,
    )

    payload_b = run_b.__dict__.copy()
    payload_b["dataset_hash"] = dataset_signature([("alt", "label")])  # force hash mismatch
    payload_b["test_ratio"] = 0.40

    ledger = build_evidence_ledger([run_a.__dict__, payload_b], accuracy_spread_threshold=0.05)
    alert_kinds = {alert["kind"] for alert in ledger["drift_alerts"]}
    assert "dataset_hash_drift" in alert_kinds
    assert "test_ratio_drift" in alert_kinds
    assert "model_dataset_inconsistency" in alert_kinds


def test_evidence_ledger_written_to_disk_and_readable(tmp_path):
    output = write_evidence_ledger(
        {
            "generated_at": "2026-01-01T00:00:00Z",
            "ledger_schema": "vp-evidence-ledger-v1",
            "run_count": 1,
            "model_count": 1,
            "runs": ["vp-s42-t25-word-frequency-10n"],
            "model_summaries": [],
            "drift_alerts": [],
        },
        tmp_path / "evidence" / "ledger.json",
    )
    assert output.exists()

    raw = output.read_text(encoding="utf-8")
    assert "vp-evidence-ledger-v1" in raw
    assert output.name == "ledger.json"


def test_build_evidence_ledger_rejects_unknown_manifest_schema():
    payload = {
        "run_id": "vp-s0-t25-word-frequency-4n",
        "model_name": "word_frequency",
        "model_type": "WordFrequencyModel",
        "seed": 1,
        "test_ratio": 0.25,
        "train_size": 3,
        "test_size": 1,
        "total_samples": 4,
        "accuracy": 0.5,
        "dataset_profile": {"label_imbalance": 0.0},
        "dataset_hash": "deadbeef",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "python_version": "3.11.0",
        "manifest_schema": "vp-manifest-v2.0",
        "parameters": {},
        "class_coverage": {},
        "labels": [],
        "confusion_matrix": {},
    }
    with pytest.raises(ValueError, match="Unsupported manifest schema version"):
        build_evidence_ledger([payload])


def test_summarize_model_group_exposes_deterministic_spread():
    entries = [
        {
            "model_name": "word_frequency",
            "model_type": "WordFrequencyModel",
            "seed": 1,
            "accuracy": 0.4,
            "dataset_hash": "aaa",
            "test_ratio": 0.3,
            "dataset_profile": {"label_distribution": {"a": 1}},
        },
        {
            "model_name": "word_frequency",
            "model_type": "WordFrequencyModel",
            "seed": 2,
            "accuracy": 0.8,
            "dataset_hash": "aaa",
            "test_ratio": 0.3,
            "dataset_profile": {"label_distribution": {"a": 1}},
        },
        {
            "model_name": "word_frequency",
            "model_type": "WordFrequencyModel",
            "seed": 3,
            "accuracy": 0.6,
            "dataset_hash": "aaa",
            "test_ratio": 0.3,
            "dataset_profile": {"label_distribution": {"a": 1}},
        },
    ]

    summary = summarize_model_group(entries)
    assert summary["run_count"] == 3
    assert summary["accuracy"]["mean"] == 0.6
    assert summary["accuracy"]["spread"] == 0.4
    assert summary["dataset_hash_stable"] is True


def test_validate_manifest_payload_accepts_schema_and_range_boundaries():
    payload = {
        "run_id": "vp-s1-t30-word-frequency-4n",
        "model_name": "word_frequency",
        "model_type": "WordFrequencyModel",
        "seed": 1,
        "test_ratio": 0.3,
        "train_size": 3,
        "test_size": 1,
        "total_samples": 4,
        "accuracy": 1.0,
        "dataset_profile": {"label_imbalance": 0.0},
        "dataset_hash": "hash",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "python_version": "3.11.0",
        "manifest_schema": "vp-manifest-v2.1",
        "parameters": {},
        "class_coverage": {},
        "labels": [],
        "confusion_matrix": {},
    }
    normalized = validate_manifest_payload(payload)
    assert normalized["manifest_schema"] == "vp-manifest-v2.1"
    assert normalized["accuracy"] == 1.0


def test_write_evidence_ledger_uses_schema_constant(tmp_path):
    output = write_evidence_ledger(
        {
            "generated_at": "2026-01-01T00:00:00Z",
            "ledger_schema": LEDGER_SCHEMA_VERSION,
            "run_count": 0,
            "model_count": 0,
            "runs": [],
            "model_summaries": [],
            "drift_alerts": [],
        },
        tmp_path / "nested" / "ledger.json",
    )
    assert output.exists()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["ledger_schema"] == LEDGER_SCHEMA_VERSION
    assert output.parent.exists()


def test_build_evidence_ledger_keeps_deterministic_run_ordering(tmp_path):
    run_a = {
        "run_id": "vp-s1-t30-word-frequency-4n",
        "model_name": "alpha",
        "model_type": "WordFrequencyModel",
        "seed": 1,
        "test_ratio": 0.3,
        "train_size": 2,
        "test_size": 2,
        "total_samples": 4,
        "accuracy": 0.75,
        "dataset_hash": "hash-a",
        "dataset_profile": {"label_distribution": {"greeting": 2}},
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "python_version": "3.11.0",
        "manifest_schema": "vp-manifest-v2.1",
        "parameters": {},
        "class_coverage": {"greeting": 2},
        "labels": ["greeting", "tech"],
        "confusion_matrix": {"greeting": {"greeting": 1, "tech": 0}, "tech": {"greeting": 1, "tech": 0}},
    }
    run_b = {
        **run_a,
        "run_id": "vp-s2-t30-word-frequency-4n",
        "seed": 2,
        "accuracy": 0.70,
    }

    ledger = build_evidence_ledger([run_b, run_a], accuracy_spread_threshold=0.01)
    assert ledger["runs"] == [run_b["run_id"], run_a["run_id"]]
    assert ledger["model_count"] == 1
    summary = ledger["model_summaries"][0]
    assert summary["accuracy"]["spread"] == 0.05
    assert summary["dataset_hash_stable"] is True
