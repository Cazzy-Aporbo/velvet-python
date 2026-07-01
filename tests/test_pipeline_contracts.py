from __future__ import annotations

from pathlib import Path

import pytest

from src.ml_pipeline import TFIDFModel
from src.pipeline import dump_run_manifests, run_classification_pipeline, run_epochs


def _tiny_dataset() -> list[tuple[str, str]]:
    return [
        ("hello friend", "greeting"),
        ("machine learning", "tech"),
        ("hello again", "greeting"),
        ("deep learning", "tech"),
        ("goodnight moon", "greeting"),
        ("cloud signals", "tech"),
        ("music and code", "tech"),
        ("welcome home", "greeting"),
    ]


def test_reproducible_manifest_payload_for_same_seed():
    dataset = _tiny_dataset()
    run_a = run_classification_pipeline(
        "tfidf",
        TFIDFModel,
        dataset,
        seed=7,
        test_ratio=0.25,
    )
    run_b = run_classification_pipeline(
        "tfidf",
        TFIDFModel,
        dataset,
        seed=7,
        test_ratio=0.25,
    )

    assert run_a.dataset_hash == run_b.dataset_hash
    assert run_a.confusion_matrix == run_b.confusion_matrix
    assert run_a.accuracy == run_b.accuracy
    assert run_a.run_id == run_b.run_id
    assert run_a.parameters == {}


def test_epoch_parameters_and_model_metadata_are_included():
    runs = run_epochs(
        "tfidf",
        TFIDFModel,
        _tiny_dataset(),
        epochs=2,
        seed=99,
    )

    assert len(runs) == 2
    assert runs[0].parameters["epoch"] == 1
    assert runs[1].parameters["epoch"] == 2
    assert runs[0].seed == 99
    assert runs[1].seed == 100
    assert "tfidf" in runs[0].model_name
    assert "model_type" in runs[0].__dict__


def test_manifest_filenames_follow_contract(tmp_path: Path):
    runs = run_epochs(
        "tfidf",
        TFIDFModel,
        _tiny_dataset(),
        epochs=1,
        seed=55,
    )
    paths = dump_run_manifests(runs, tmp_path / "manifests")
    assert len(paths) == 1

    manifest_path = paths[0]
    assert manifest_path.name.endswith(".json")
    assert manifest_path.parent == tmp_path / "manifests"
    assert manifest_path.exists()

    run = runs[0]
    assert manifest_path.name.startswith(run.run_id.replace(" ", "_"))


def test_run_classification_pipeline_requires_train_and_predict_methods() -> None:
    class IncompleteModel:
        pass

    with pytest.raises(TypeError, match="train and predict"):
        run_classification_pipeline(
            "incomplete",
            IncompleteModel,
            [("hello", "greeting"), ("goodbye", "casual")],
            seed=1,
        )


def test_run_classification_pipeline_requires_train_and_predict_on_separate_implementations() -> None:
    class MissingTrain:
        def predict(self, text: str) -> str:
            return "x"

    class MissingPredict:
        def train(self, texts: list[str], labels: list[str]) -> None:
            return None

    with pytest.raises(TypeError, match="train and predict"):
        run_classification_pipeline(
            "missing-train",
            MissingTrain,
            [("hello", "greeting"), ("goodbye", "casual")],
            seed=2,
        )

    with pytest.raises(TypeError, match="train and predict"):
        run_classification_pipeline(
            "missing-predict",
            MissingPredict,
            [("hello", "greeting"), ("goodbye", "casual")],
            seed=2,
        )


def test_run_classification_pipeline_rejects_empty_input_dataset() -> None:
    with pytest.raises(ValueError, match="No usable rows"):
        run_classification_pipeline(
            "tfidf",
            TFIDFModel,
            [],
        )


def test_run_classification_pipeline_keeps_run_parameters_and_data_profile() -> None:
    dataset = [
        ("hello friend", "greeting"),
        ("machine learning", "tech"),
        ("hello again", "greeting"),
        ("deep learning", "tech"),
    ]
    run = run_classification_pipeline(
        "tfidf",
        TFIDFModel,
        dataset,
        seed=21,
        parameters={"feature_focus": "tfidf", "notes": "contract test"},
    )

    assert run.parameters["feature_focus"] == "tfidf"
    assert run.parameters["notes"] == "contract test"
    assert run.dataset_profile["total_records"] == len(dataset)
    assert run.dataset_profile["label_count"] == 2


def test_run_classification_pipeline_rejects_non_callable_model_builder() -> None:
    with pytest.raises(TypeError):
        run_classification_pipeline(
            "bad",
            123,  # type: ignore[arg-type]
            [("hello", "greeting"), ("goodbye", "casual")],
        )
