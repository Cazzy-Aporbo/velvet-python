<h1 align="center" style="color:#9d7fbf;">velvet-python</h1>

<p align="center" style="color:#74608a;">
From clean examples to reproducible systems.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-F4DEFF?style=for-the-badge&logo=python&logoColor=5B4B79&labelColor=D8C5F0" alt="Python"/>
  <img src="https://img.shields.io/badge/Tests-Pytest-F2E2FF?style=for-the-badge&logo=pytest&logoColor=4B3B66&labelColor=E8D4F4" alt="Pytest"/>
  <img src="https://img.shields.io/badge/Quality-Ruff-F8E5FF?style=for-the-badge&logo=ruff&logoColor=5A4B6C&labelColor=EED9F7" alt="Ruff"/>
  <img src="https://img.shields.io/badge/License-MIT-EED9FF?style=for-the-badge&logo=open-source&labelColor=DCC2EE" alt="License"/>
</p>

## What this repository is for

`velvet-python` is a teaching-oriented but production-minded project where code, tests, and CLI outputs are part of the curriculum.

It is designed around one principle:

- **Every useful claim has a reproducible artifact**

You learn by reading code, running the same experiment twice, and checking whether your results match the stored manifests.

## Why this is different

Most learning repositories end at “how to make it run.”

This one keeps going until the behavior is explainable:

- deterministic dataset contracts in `src/data_utils.py`
- reproducible train/test splits in `src/pipeline.py`
- model sweep automation in `scripts/run_experiments.py`
- dataset audits for input quality in `scripts/dataset_audit.py`
- tests that pin expected behavior in `tests/`

The result is a practical path for:

- students who want stronger habits than random snippets,
- data engineers who need to keep assumptions explicit,
- interview candidates who want to explain tradeoffs under pressure.

## Repository map

```text
velvet-python/
├─ src/
│  ├─ ai.py              Rule/Bayes/Cosine classifiers
│  ├─ data_utils.py      Dataset loading and deterministic validation
│  ├─ ml_pipeline.py     Feature + model wrappers
│  ├─ pipeline.py        Experiment orchestration and manifests
│  └─ __init__.py
├─ scripts/
│  ├─ run_experiments.py  CLI baseline sweep + JSON/CVS outputs
│  ├─ dataset_audit.py    Data health checks and hashable evidence
│  └─ other demos for exploration
├─ tests/
│  ├─ pipeline and data utility tests
│  └─ CLI and benchmark smoke checks
├─ docs/
│  ├─ learning-path.md
│  ├─ experiment-ledger.md
│  └─ practice-labs.md
├─ pyfiles/               Reference implementations and exercises
├─ makefile              Quality and onboarding commands
└─ README.md
```

## Core engineering ideas (in practice)

### 1) Deterministic data and fast failure

`load_dataset`, `load_csv`, and `validate_dataset` enforce structure before any model sees data:

- every row must be `(text, label)`
- empty text/labels fail immediately
- duplicated, missing, malformed, and malformed header cases are explicit

No implicit coercion is the goal. If a dataset is wrong, you want to know early.

### 2) Reproducible experiment runs

`run_classification_pipeline` returns an `ExperimentRun` with stable metadata:

- `seed`, `test_ratio`, and split sizes
- split results, accuracy, class coverage
- confusion matrix and dataset hash
- run-specific `parameters`

This makes model choices defensible and reviewable.

### 3) Evidence-first outputs

Every non-trivial script writes machine-readable outputs:

- JSON manifests per run (`artifacts/*.json`)
- optional `summary.csv` for spreadsheet comparison
- optional dataset audit report (`dataset_audit.py --write-json`)

If you cannot compare run artifacts over time, you are guessing.

## Quick start (10-minute onboarding)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### First quality checks

```bash
make install
make check
```

### Run one learning sweep

```bash
python scripts/run_experiments.py --epochs 2 --seed 42
```

Artifacts are written to `artifacts/`.

### Generate a dataset audit profile

```bash
python scripts/dataset_audit.py --source builtin --batch-size 2 --write-json artifacts/dataset.json
```

### Run the CLI pipeline from Typer

```bash
python CLI.py pipeline --epochs 2 --seed 44 --output artifacts
```

## How to interpret outputs (so you can trust results)

### Manifest fields (baseline)

- `run_id`, `started_at`, `finished_at`
- `model_name`, `model_type`
- `seed`, `test_ratio`, `train_size`, `test_size`
- `accuracy`, `confusion_matrix`, `class_coverage`
- `dataset_profile` (quality signals like duplicate rate and label imbalance)
- `dataset_hash`

### Practical interpretation rules

- **Accuracy changes while split is fixed?** Check model behavior and preprocessing first.
- **Confusion gets worse but accuracy looks stable?** Verify class balance and label coverage.
- **`dataset_hash` changes without intent?** Freeze upstream corpus source and rerun from scratch.

Use these checks before any “improvement” claim.

## Learning by role

| Role | First pass | Second pass |
|---|---|---|
| Beginner developer | `CLI.py pipeline`, `src/data_utils.py` | `tests/test_data_pipeline_utils.py`, `test_run_experiments_cli.py` |
| Data engineer | `dataset_audit.py`, split strategy + hash checks | add a malformed CSV case to tests and assert failure behavior |
| ML engineer | `run_epochs`, builder registry, manifest schema | compare model runs across seeds and record variance |
| Interview prep | failure patterns in `docs/experiment-ledger.md` | explain one run where variance improves accuracy but hurts recall |

## Practical workflow for contributors

1. Add or refactor one change.
2. Add/extend one test that captures behavior.
3. Run:

```bash
make lint
autotest=$(make test-fast)
```

4. Re-run the relevant workflow and commit manifest evidence in your PR notes.

## Added utility: audit + lab-oriented scripts

### `scripts/run_experiments.py`

- adds model filters with `--models`
- stores one manifest per model x epoch
- optional `--summary-csv` for easy review

### `scripts/dataset_audit.py`

- computes dataset quality profile
- supports builtin dataset and CSV input
- optional preview of batch extraction
- optional JSON report for auditability

### `docs/practice-labs.md`

- staged practical exercises
- role-specific prompts
- evidence checklists for each task

## Testing strategy

Use one command for the gate:

```bash
make check
```

For deeper evidence before major changes, use the focused verification set:

```bash
pytest \
  tests/test_data_pipeline_utils.py \
  tests/test_dataset_audit_cli.py \
  tests/test_run_experiments_cli.py \
  tests/test_algorithm_guide_registry.py \
  tests/test_data_utils_and_ingestion.py \
  tests/test_cli_recommend_command.py
```

Current test scope:

- `tests/test_data_pipeline_utils.py` — data contracts and batching
- `tests/test_run_experiments_cli.py` — reproducible scripts and outputs
- `tests/test_dataset_audit_cli.py` — dataset health CLI behavior
- `tests/test_algorithm_guide_registry.py` — recommendation + registry consistency
- `tests/test_data_utils_and_ingestion.py` — CSV/data integrity and hash invariants
- `tests/test_cli_recommend_command.py` — recommendation flow from the command line
- core baseline and integration tests in existing suite

## Roadmap (pragmatic)

### 1–2 days

- add two small model variants in a safe way
- expand bad-input cases in dataset loading tests
- tighten summary reporting defaults

### 1–2 weeks

- add `schema_version` and loader provenance to manifests
- include replayability checks (rerun and diff hash behavior)
- add a small benchmarking guide for CPU/runtime drift

### 1–3 months

- add lightweight streaming demo input using fixed-size batches
- grow the lab docs into topic tracks for fairness, observability, and release-readiness
- publish one reference “engineering playbook” PR example with evidence

## Next read

- [docs/learning-path.md](docs/learning-path.md)
- [docs/experiment-ledger.md](docs/experiment-ledger.md)
- [docs/practice-labs.md](docs/practice-labs.md)
- [contributions.md](contributions.md)

## License

MIT for source code.

## Attribution

Built and maintained by [Cazzy Aporbo](https://github.com/Cazzy-Aporbo).
