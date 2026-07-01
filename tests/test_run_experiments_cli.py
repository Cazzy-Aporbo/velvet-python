from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.run_experiments as runner


def test_run_experiments_outputs_manifests_and_summary(monkeypatch, tmp_path, capsys):
    """End-to-end experiment script should emit manifests and a machine-readable summary."""

    dataset = [
        ("hello world", "greeting"),
        ("machine learning workflows", "tech"),
        ("good morning", "greeting"),
        ("deep learning models", "tech"),
        ("the weather", "casual"),
        ("coffee and code", "casual"),
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda: dataset)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments",
            "--epochs",
            "2",
            "--seed",
            "13",
            "--output-dir",
            str(tmp_path),
        ],
    )

    runner.main()
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert "summary" in payload
    assert "manifest_files" in payload
    assert Path(payload["output_dir"]) == tmp_path.resolve()

    # 4 models * 2 epochs.
    assert len(payload["summary"]) == 8
    assert len(payload["manifest_files"]) == 8

    for filename in payload["manifest_files"]:
        file_path = tmp_path / filename
        assert file_path.exists()
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        assert raw["total_samples"] == len(dataset)
        assert raw["test_ratio"] == 0.30
        assert "dataset_profile" in raw
        assert 0.0 <= raw["accuracy"] <= 1.0


def test_run_experiments_model_filter_and_csv(monkeypatch, tmp_path, capsys):
    """Selective model runs should work and produce a CSV summary artifact."""

    dataset = [
        ("hello world", "greeting"),
        ("machine learning workflows", "tech"),
        ("good morning", "greeting"),
        ("deep learning models", "tech"),
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda: dataset)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments",
            "--models",
            "word_frequency,tfidf",
            "--epochs",
            "1",
            "--summary-csv",
            "--output-dir",
            str(tmp_path),
        ],
    )

    runner.main()
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert payload["summary_csv"] == "summary.csv"
    assert len(payload["manifest_files"]) == 2

    summary_csv = tmp_path / payload["summary_csv"]
    assert summary_csv.exists()
    csv_text = summary_csv.read_text(encoding="utf-8")
    rows = [line for line in csv_text.splitlines() if line.strip()]
    # header + two model runs
    assert len(rows) == 3


def test_run_experiments_can_write_ledger(monkeypatch, tmp_path, capsys):
    dataset = [
        ("hello world", "greeting"),
        ("machine learning workflows", "tech"),
        ("good morning", "greeting"),
        ("deep learning models", "tech"),
        ("the weather", "casual"),
        ("coffee and code", "casual"),
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda: dataset)

    ledger_path = tmp_path / "evidence" / "ledger.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments",
            "--models",
            "word_frequency",
            "--epochs",
            "1",
            "--ledger",
            str(ledger_path),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ],
    )

    runner.main()
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert payload["ledger_path"] == str(ledger_path)
    assert "evidence" in payload
    assert payload["evidence"]["drift_count"] >= 0
    assert payload["evidence"]["ledger_schema"] == "vp-evidence-ledger-v1"
    assert ledger_path.exists()
    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert raw["model_count"] == 1
    assert raw["ledger_schema"] == "vp-evidence-ledger-v1"


def test_run_experiments_rejects_unknown_model_names(monkeypatch, tmp_path):
    dataset = [
        ("hello world", "greeting"),
        ("machine learning workflows", "tech"),
        ("good morning", "greeting"),
        ("deep learning models", "tech"),
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda: dataset)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments",
            "--models",
            "word_frequency,does_not_exist",
            "--epochs",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="Unknown model"):
        runner.main()
