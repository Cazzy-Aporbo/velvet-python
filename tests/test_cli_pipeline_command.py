from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cli_test_utils import run_cli


def test_pipeline_command_generates_manifests(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"

    result = run_cli(
        [
            "pipeline",
            "--epochs",
            "1",
            "--seed",
            "123",
            "--output",
            str(output_dir),
        ]
    )

    assert result.return_code == 0
    output = result.stdout + result.stderr
    assert "Pipeline run complete" in output
    assert output_dir.exists()
    files = sorted(p for p in output_dir.glob("vp-s123-t30-*.json"))
    assert len(files) == 4

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert "run_id" in payload
    assert payload["seed"] == 123
    assert payload["manifest_schema"].startswith("vp-manifest-")
    assert "dataset_profile" in payload
    assert "python_version" in payload


@pytest.mark.parametrize(
    "invalid_option,error_fragment",
    [
        (["--epochs", "0"], "invalid value"),
        (["--seed", "abc"], "invalid value"),
    ],
)
def test_pipeline_command_rejects_invalid_options(
    invalid_option: list[str], error_fragment: str, tmp_path: Path
) -> None:
    output_dir = tmp_path / "bad"
    result = run_cli(
        ["pipeline", *invalid_option, "--output", str(output_dir)],
    )
    assert result.return_code != 0
    output = result.stdout + result.stderr
    assert error_fragment.lower() in output.lower()


def test_pipeline_command_writes_evidence_ledger(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    ledger_path = tmp_path / "evidence.json"

    result = run_cli(
        [
            "pipeline",
            "--epochs",
            "1",
            "--seed",
            "21",
            "--output",
            str(output_dir),
            "--ledger",
            str(ledger_path),
        ],
    )

    assert result.return_code == 0
    assert output_dir.exists()
    assert ledger_path.exists()

    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert raw["ledger_schema"] == "vp-evidence-ledger-v1"
    assert raw["run_count"] == 4
    assert raw["model_count"] == 4
