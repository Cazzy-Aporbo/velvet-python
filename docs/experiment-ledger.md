# Experiment Ledger and Reproducibility Guide

<h2 style="color:#8f73b4;">Purpose</h2>

This repo treats each model run as ledger evidence.
Each run generates one JSON manifest and is auditable by inspection.

We keep this strict because engineering progress only matters when it is
verifiable:

- no hidden assumptions,
- no untracked baseline drift,
- no claims without evidence.

<h2 style="color:#8f73b4;">What is captured in one run</h2>

Every run stores:

- `run_id`, `started_at`, `finished_at`
- model identity (`model_name`, `model_type`)
- deterministic controls (`seed`, `test_ratio`)
- sample sizes and labels
- `accuracy`, `confusion_matrix`, and `class_coverage`
- stable `dataset_hash`
- run `parameters` for traceability
- `dataset_profile` with duplication and length signals

<h2 style="color:#8f73b4;">How to interpret the ledger</h2>

Use this sequence before changing architecture:

1. Run a baseline sweep and save manifest outputs.
2. Compare runs with fixed `seed` and fixed dataset.
3. Vary seed only for variance checks.
4. Capture `parameters` whenever data handling or preprocessing changes.

<h2 style="color:#8f73b4;">Run protocols</h2>

### Baseline protocol (onboarding)

```bash
python scripts/run_experiments.py --epochs 3 --seed 42 --test-ratio 0.30
```

### Evidence protocol (reproducible review packets)

```bash
python scripts/run_experiments.py --epochs 2 --seed 42 --ledger artifacts/evidence-ledger.json
```

```bash
python CLI.py pipeline --epochs 2 --seed 21 --ledger artifacts/pipeline-ledger.json
```

These commands produce:

- per-model JSON manifests
- optional `summary.csv` when requested
- `evidence-ledger` JSON containing:
  - model-level summary tables
  - reproducibility drift alerts
  - deterministic checks for split and dataset hash stability

### Stability protocol (dataset/process changes)

```bash
make all-runs
```

Then compare manifest `accuracy`, `confusion_matrix`, and `dataset_hash` before merge.

### Decision protocol

Only accept a change when:

- baseline behavior is unchanged where expected,
- class-level changes are intentional,
- no critical validation logic is weakened without explicit review.

<h2 style="color:#8f73b4;">How we test for integrity</h2>

The integrity tests in:

- `tests/test_data_pipeline_utils.py`
- `tests/test_dataset_audit_cli.py`
- `tests/test_run_experiments_cli.py`
- `tests/test_algorithm_guide_registry.py`
- `tests/test_data_utils_and_ingestion.py`
- `tests/test_cli_recommend_command.py`

cover malformed rows, deterministic split/manifest behavior, and CLI failure modes.

Run:

```bash
pytest \
  tests/test_data_pipeline_utils.py \
  tests/test_dataset_audit_cli.py \
  tests/test_run_experiments_cli.py \
  tests/test_algorithm_guide_registry.py \
  tests/test_data_utils_and_ingestion.py \
  tests/test_cli_recommend_command.py
```

<h2 style="color:#8f73b4;">Checklist before opening a PR</h2>

- `python -m ruff check src tests`
- `pytest` command above (focused evidence set)
- `python scripts/run_experiments.py --epochs 2 --seed 42 --summary-csv`
- `python scripts/dataset_audit.py --source builtin --write-json artifacts/dataset-audit.json`
- include manifest artifact references in PR notes
- `python CLI.py recommend --rows 240 --classes 3 --probabilities` (manual review of tradeoff ranking)

<h2 style="color:#8f73b4;">Failure patterns to avoid</h2>

### Pattern: changing accuracy with no manifest delta

This usually means hidden preprocessing drift.
Require explicit comparison and a written rationale for any drift.

### Pattern: better average but worse minority classes

Use `confusion_matrix` and per-class recall to confirm whether gains come with hidden harm.

### Pattern: test suite passes but behavior changed

Check:

- `dataset_hash`
- train/test split controls
- model builder registration in `CLI.py` and `scripts/run_experiments.py`
- whether `parameters` were updated for data/model assumptions.
