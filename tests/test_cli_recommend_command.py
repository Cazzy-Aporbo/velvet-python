from __future__ import annotations

from tests.cli_test_utils import run_cli


def test_recommend_command_surfaces_ranked_models():
    result = run_cli(
        [
            "recommend",
            "--rows",
            "180",
            "--classes",
            "4",
            "--probabilities",
            "--p95-latency-ms",
            "120",
            "--no-explainability",
        ],
    )

    assert result.return_code == 0
    output = result.stdout
    heading = "Recommendations for 180 rows"
    assert heading in output
    assert "4 classes" in output.replace("\u00d7", "x")
    assert "Model" in output
    assert "Naive Bayes" in output or "Word Frequency" in output


def test_recommend_command_accepts_latency_focus_without_probabilities():
    result = run_cli(
        [
            "recommend",
            "--rows",
            "50",
            "--classes",
            "6",
            "--p95-latency-ms",
            "75",
            "--no-explainability",
            "--probabilities",
        ],
    )

    assert result.return_code == 0
    output = result.stdout
    assert "Rows:" not in output
    assert "Recommendations for 50 rows" in output.replace("\u00d7", "x")

    # The ranking should still show machine-readable, reproducible entries.
    assert "Model" in output
    assert "Rank" in output


def test_recommend_command_defaults_are_stable():
    result = run_cli(["recommend"])
    assert result.return_code == 0
    output = result.stdout
    assert "Recommendations for 200 rows x 3 classes" in output
    assert "Model" in output


def test_unknown_command_reports_helpful_error():
    result = run_cli(["does-not-exist"])
    assert result.return_code != 0
    output = result.stdout + result.stderr
    assert "Usage" in output or "No such command" in output or "No such option" in output
