from __future__ import annotations

import json
import sys

from src.data_utils import dataset_signature, load_dataset
from src.ml_pipeline import WordFrequencyModel
from src.pipeline import (
    MANIFEST_SCHEMA_VERSION,
    dump_run_manifest,
    run_classification_pipeline,
)


def test_manifest_contains_stable_evidence_fields(tmp_path):
    dataset = load_dataset()
    run = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=17,
        test_ratio=0.3,
        parameters={"note": "manifest_contract"},
    )

    path = dump_run_manifest(run, tmp_path / "run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["model_name"] == "word_frequency"
    expected_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    assert payload["python_version"] == expected_python
    assert payload["manifest_schema"] == MANIFEST_SCHEMA_VERSION
    assert payload["dataset_hash"] == dataset_signature(dataset)
    assert "dataset_profile" in payload
    assert payload["dataset_profile"]["label_imbalance"] >= 0.0
    assert "confusion_matrix" in payload
    assert "run_id" in payload


def test_reproducible_runs_have_identical_scientific_artifacts():
    dataset = load_dataset()

    run_a = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=99,
    )
    run_b = run_classification_pipeline(
        model_name="word_frequency",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=99,
    )

    assert run_a.accuracy == run_b.accuracy
    assert run_a.dataset_hash == run_b.dataset_hash
    assert run_a.confusion_matrix == run_b.confusion_matrix
    assert run_a.seed == run_b.seed
