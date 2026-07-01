"""Command-line utility for practical dataset checks and reproducibility evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from src.data_utils import (
    dataset_profile,
    dataset_signature,
    iter_batches,
    load_csv,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect dataset quality and emit a reproducible profile for "
            "local experiments and PR reviews."
        )
    )
    parser.add_argument(
        "--source",
        default="builtin",
        choices=("builtin", "csv"),
        help="Data source: builtin in-code corpus or CSV input",
    )
    parser.add_argument("--csv", dest="csv_path", help="Path to CSV file when source=csv")
    parser.add_argument("--text-col", default="text", help="CSV text column name")
    parser.add_argument("--label-col", default="label", help="CSV label column name")
    parser.add_argument("--batch-size", type=int, default=0, help="Optional batch-size preview")
    parser.add_argument("--preview-batches", type=int, default=2, help="Batch count in preview")
    parser.add_argument(
        "--write-json",
        help="Optional path to persist the audit report as JSON",
    )
    return parser.parse_args()


def build_audit_report(
    data: list[tuple[str, str]],
    *,
    source: str,
    source_path: str | None = None,
    batch_size: int = 0,
    preview_batches: int = 2,
) -> dict:
    """Build a complete audit payload for one dataset."""
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    profile = dataset_profile(data)
    report = {
        "source": source,
        "source_path": source_path,
        "sample_count": len(data),
        "profile": profile,
        "dataset_hash": dataset_signature(data),
    }

    if batch_size > 0:
        preview = []
        for idx, batch in enumerate(iter_batches(data, batch_size=batch_size)):
            if idx >= preview_batches:
                break
            preview.append({"batch_index": idx, "rows": batch})
        report["batch_preview"] = preview
        report["batch_size"] = batch_size

    return report


def main() -> None:
    args = parse_args()
    if args.source == "csv":
        if not args.csv_path:
            raise ValueError("csv path is required when source=csv")
        data = load_csv(
            args.csv_path,
            text_col=args.text_col,
            label_col=args.label_col,
        )
        source_path = str(Path(args.csv_path))
    else:
        data = load_dataset()
        source_path = None

    report = build_audit_report(
        data,
        source=args.source,
        source_path=source_path,
        batch_size=args.batch_size,
        preview_batches=args.preview_batches,
    )

    if args.write_json:
        Path(args.write_json).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"status": "written", "path": str(Path(args.write_json))}, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
