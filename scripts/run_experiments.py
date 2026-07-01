"""Run educational, reproducible model experiments from the command line."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

from src.data_utils import load_dataset
from src.evidence_ledger import build_evidence_ledger, write_evidence_ledger
from src.model_registry import model_builders, model_labels, model_metadata, sorted_model_keys
from src.pipeline import (
    canonical_model_name,
    dump_run_manifests,
    run_epochs,
    summarize_run_series,
    summarize_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible baseline classifiers and emit JSON run manifests "
            "for learning and benchmark review."
        )
    )
    parser.add_argument("--output-dir", default="artifacts", help="Manifest output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch count for every model")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for splits")
    parser.add_argument("--test-ratio", type=float, default=0.30, help="Validation holdout ratio")
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated model keys to run (default: all). "
            f"Available: {', '.join(sorted_model_keys())}"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        action="store_true",
        help="Also write summary.csv in the output directory",
    )
    parser.add_argument(
        "--ledger",
        default="",
        help=(
            "Optional path to write a reproducible evidence ledger. "
            "When provided, all run manifests are aggregated into a single JSON summary."
        ),
    )
    return parser.parse_args()


def build_models() -> dict[str, Callable]:
    return model_builders()


def write_summary_csv(manifest_paths: list[Path], output_dir: Path) -> Path:
    """Persist tabular run summaries for spreadsheet consumers."""
    rows = []
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(payload)

    if not rows:
        return output_dir / "summary.csv"

    csv_fields = [
        "run_id",
        "base_model_name",
        "model_name",
        "model_type",
        "seed",
        "test_ratio",
        "train_size",
        "test_size",
        "accuracy",
        "run_duration_seconds",
        "total_records",
        "label_imbalance",
        "dataset_hash",
    ]
    output = output_dir / "summary.csv"
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)
        for row in rows:
            profile = row.get("dataset_profile", {})
            writer.writerow(
                [
                    row.get("run_id", ""),
                    canonical_model_name(row.get("model_name", "")),
                    row.get("model_name", ""),
                    row.get("model_type", ""),
                    row.get("seed", ""),
                    row.get("test_ratio", ""),
                    row.get("train_size", ""),
                    row.get("test_size", ""),
                    row.get("accuracy", ""),
                    row.get("run_duration_seconds", ""),
                    profile.get("total_records", ""),
                    profile.get("label_imbalance", ""),
                    row.get("dataset_hash", ""),
                ],
            )

    return output


def main() -> None:
    args = parse_args()
    dataset = load_dataset()

    selected = {name.strip().lower() for name in args.models.split(",") if name.strip()}
    available = build_models()
    labels = model_labels()
    metadata = model_metadata()
    model_names = sorted(available) if args.models == "all" or not selected else sorted(selected)

    all_runs = []
    for name in model_names:
        builder = available.get(name)
        if not builder:
            raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(available))}")
        runs = run_epochs(
            model_name=name,
            model_builder=builder,
            data=dataset,
            epochs=args.epochs,
            seed=args.seed,
            test_ratio=args.test_ratio,
            parameters={"pipeline": "baseline", "source": "load_dataset()"},
        )
        all_runs.extend(runs)

    output = Path(args.output_dir)
    paths = dump_run_manifests(all_runs, output)
    summary = summarize_runs(all_runs)
    series_summary = summarize_run_series(all_runs)
    ledger_payload = None
    ledger_path = None

    if args.ledger:
        ledger_payload = build_evidence_ledger(
            all_runs,
            accuracy_spread_threshold=0.05,
        )
        ledger_path = write_evidence_ledger(ledger_payload, args.ledger)

    summary_csv = None
    if args.summary_csv:
        summary_csv = write_summary_csv(paths, output)

    report = {
        "output_dir": str(output.resolve()),
        "manifest_files": [str(path.name) for path in paths],
        "summary": summary,
        "series_summary": series_summary,
        "selected_models": [
            {
                "key": key,
                "label": labels[key],
                "metadata": metadata[key],
            }
            for key in model_names
        ],
    }
    if summary_csv is not None:
        report["summary_csv"] = str(summary_csv.name)
    if ledger_path is not None:
        report["ledger_path"] = str(Path(ledger_path))
        report["evidence"] = {
            "ledger_schema": ledger_payload["ledger_schema"],
            "drift_count": len(ledger_payload["drift_alerts"]),
            "model_count": ledger_payload["model_count"],
            "health": ledger_payload["health"],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
