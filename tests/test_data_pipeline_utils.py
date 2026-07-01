from __future__ import annotations

from pathlib import Path

import pytest

from src.data_utils import dataset_profile, iter_batches, validate_dataset
from src.ml_pipeline import WordFrequencyModel
from src.pipeline import (
    dump_run_manifests,
    run_classification_pipeline,
    run_epochs,
    summarize_runs,
)


def test_validate_dataset_rejects_bad_shape():
    bad = ["not-a-row", ("text",), (1, "label")]
    with pytest.raises(ValueError):
        validate_dataset(bad)  # type: ignore[arg-type]


def test_validate_dataset_removes_whitespace_and_keeps_labels():
    data = [(" hello ", " greeting "), ("tech \n", " tech")]
    normalized = validate_dataset(data)
    assert normalized[0] == ("hello", "greeting")
    assert normalized[1] == ("tech", "tech")


def test_validate_dataset_rejects_empty_records():
    with pytest.raises(ValueError, match="empty text"):
        validate_dataset([("", "label")])


def test_run_classification_pipeline_returns_manifest():
    dataset = [
        ("hello friend", "greeting"),
        ("machine learning", "tech"),
        ("hello again", "greeting"),
        ("deep learning", "tech"),
    ]
    run = run_classification_pipeline(
        model_name="wordfreq",
        model_builder=WordFrequencyModel,
        data=dataset,
        seed=13,
        test_ratio=0.5,
    )

    assert run.model_name == "wordfreq"
    assert 0.0 <= run.accuracy <= 1.0
    assert run.total_samples == len(dataset)
    assert isinstance(run.confusion_matrix, dict)
    assert run.parameters == {}


def test_run_classification_pipeline_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="test_ratio"):
        run_classification_pipeline(
            model_name="invalid",
            model_builder=WordFrequencyModel,
            data=[("hello", "greeting"), ("machine", "tech")],
            test_ratio=1.5,
        )


def test_run_epochs_is_reproducible_for_fixed_seed():
    dataset = [
        ("hello friend", "greeting"),
        ("machine learning", "tech"),
        ("hello again", "greeting"),
        ("deep learning", "tech"),
    ]
    first = run_epochs("wordfreq", WordFrequencyModel, dataset, epochs=2, seed=99)
    second = run_epochs("wordfreq", WordFrequencyModel, dataset, epochs=2, seed=99)
    assert [r.accuracy for r in first] == [r.accuracy for r in second]
    assert first[0].seed != first[1].seed


def test_run_epochs_rejects_non_positive_epochs():
    with pytest.raises(ValueError, match="epochs"):
        run_epochs(
            "wordfreq",
            WordFrequencyModel,
            [("a", "x"), ("b", "y")],
            epochs=0,
        )


def test_dump_and_summarize_runs(tmp_path: Path):
    dataset = [
        ("hello friend", "greeting"),
        ("machine learning", "tech"),
        ("hello again", "greeting"),
        ("deep learning", "tech"),
    ]
    runs = run_epochs("wordfreq", WordFrequencyModel, dataset, epochs=2, seed=11)
    files = dump_run_manifests(runs, tmp_path / "manifests")

    assert len(files) == 2
    assert all(path.exists() for path in files)

    summary = summarize_runs(runs)
    assert len(summary) == 2
    assert all({"model_name", "seed", "accuracy", "train_size", "test_size"} <= row.keys() for row in summary)


def test_dataset_profile_contains_quality_signals():
    data = [
        ("hello", "greeting"),
        ("hello", "greeting"),
        ("hello", "tech"),
        ("world", "tech"),
        ("world", "tech"),
        ("data", "tech"),
    ]
    profile = dataset_profile(data)

    assert profile["total_records"] == 6
    assert profile["unique_records"] == 4
    assert profile["duplicate_records"] == 2
    assert profile["unique_texts"] == 3
    assert profile["duplicated_texts"] == 2
    assert profile["label_count"] == 2
    assert 0.0 <= profile["label_imbalance"] <= 1.0


def test_iter_batches_streaming_contract():
    data = [("hello", "greeting"), ("hello2", "tech"), ("hello3", "tech"), ("hello4", "casual")]
    assert list(iter_batches(data, batch_size=2)) == [
        [("hello", "greeting"), ("hello2", "tech")],
        [("hello3", "tech"), ("hello4", "casual")],
    ]


def test_iter_batches_rejects_bad_size():
    data = [("hello", "greeting")]
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(iter_batches(data, batch_size=0))
