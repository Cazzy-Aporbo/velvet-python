from __future__ import annotations

import json
import sys

import pytest

import scripts.dataset_audit as audit


def test_dataset_audit_builtin_emits_profile(monkeypatch, tmp_path, capsys):
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_audit",
            "--source",
            "builtin",
            "--batch-size",
            "2",
            "--write-json",
            str(output),
        ],
    )

    audit.main()
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)

    assert payload["status"] == "written"
    assert payload["path"] == str(output)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["source"] == "builtin"
    assert report["sample_count"] > 0
    assert "profile" in report
    assert report["batch_size"] == 2
    assert report["profile"]["label_count"] >= 2
    assert len(report["batch_preview"]) == 2
    assert len(report["batch_preview"][0]["rows"]) == 2


def test_dataset_audit_csv_rejects_missing_path(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_audit",
            "--source",
            "csv",
        ],
    )
    try:
        audit.main()
    except ValueError as exc:
        assert "csv path is required" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError when csv path is missing")


def test_dataset_audit_csv_rejects_missing_columns(tmp_path, monkeypatch):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("wrong_name,label\nhello,world\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_audit",
            "--source",
            "csv",
            "--csv",
            str(csv_path),
            "--text-col",
            "text",
        ],
    )

    with pytest.raises(KeyError, match="Missing columns"):
        audit.main()


def test_dataset_audit_preview_batch_controls_work_like_contract():
    """Preview batching should appear only when requested and match batch_size exactly."""
    data = [("hello", "greeting"), ("tech", "tech"), ("news", "news")]
    report = audit.build_audit_report(
        data,
        source="builtin",
        batch_size=2,
        preview_batches=1,
    )

    assert report["batch_size"] == 2
    assert report["batch_preview"][0]["batch_index"] == 0
    assert report["batch_preview"][0]["rows"] == [("hello", "greeting"), ("tech", "tech")]
    assert len(report["batch_preview"]) == 1


def test_dataset_audit_rejects_negative_batch_sizes():
    with pytest.raises(ValueError, match="batch_size must be non-negative"):
        audit.build_audit_report(
            [("hello", "greeting")],
            source="builtin",
            batch_size=-1,
        )
