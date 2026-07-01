from __future__ import annotations

from pathlib import Path

import pytest

from src.data_utils import (
    dataset_profile,
    dataset_signature,
    load_csv,
    validate_dataset,
)


def test_dataset_signature_is_order_invariant_and_sensitive():
    """Dataset hashes should ignore ordering but respond to content change."""
    baseline = [("a", "x"), ("b", "y"), ("a", "x")]
    reversed_order = list(reversed(baseline))

    assert dataset_signature(baseline) == dataset_signature(reversed_order)

    altered = [("a", "x"), ("b", "z"), ("a", "x")]
    assert dataset_signature(baseline) != dataset_signature(altered)


def test_validate_dataset_trims_and_preserves_labels():
    data = [(" hello ", " Greeting "), ("", "x")]
    with pytest.raises(ValueError, match="empty text"):
        validate_dataset(data)

    normalized = validate_dataset([(" hello ", " Greeting "), ("WORLD", "x")])
    assert normalized == [("hello", "Greeting"), ("WORLD", "x")]


def test_load_csv_parses_text_and_label_columns(tmp_path: Path):
    """Malformed rows should be dropped while preserving schema validity."""
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        (
            "text,label,ignored\n"
            "hello world,greeting,ignore\n"
            ",bad,row\n"
            "machine learning,tech,ignore\n"
            "  ,tech,ignore\n"
        ),
        encoding="utf-8",
    )

    rows = load_csv(csv_path, text_col="text", label_col="label")
    assert rows == [("hello world", "greeting"), ("machine learning", "tech")]


def test_load_csv_rejects_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("text_only\nvalue,label\n", encoding="utf-8")

    with pytest.raises(KeyError, match="Missing columns"):
        load_csv(csv_path, text_col="text", label_col="label")


def test_load_csv_rejects_headerless_file(tmp_path: Path):
    csv_path = tmp_path / "headerless.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(KeyError, match="no header row"):
        load_csv(csv_path, text_col="text", label_col="label")


def test_dataset_profile_distribution_and_imbalance():
    profile = dataset_profile([("a", "x"), ("b", "x"), ("a", "y"), ("c", "y"), ("d", "y")])
    assert profile["total_records"] == 5
    assert profile["unique_records"] == 5
    assert profile["label_distribution"] == {"x": 2, "y": 3}
    assert profile["label_imbalance"] == 0.029
