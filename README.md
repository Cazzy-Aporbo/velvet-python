<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,16,18,20,22&height=240&section=header&text=velvet-python&fontSize=72&animation=fadeIn&fontAlignY=36&desc=Code%20you%20can%20read,%20rerun,%20and%20learn%20from%20slowly&descAlignY=57&descSize=20&fontColor=6D5F80" />

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-F6E6FF?style=for-the-badge&logo=python&logoColor=5A4B73&labelColor=E6D2F5" alt="Python"/>
  <img src="https://img.shields.io/badge/Tests-Pytest-FFE6F3?style=for-the-badge&logo=pytest&logoColor=5A4B73&labelColor=F2D6EA" alt="Pytest"/>
  <img src="https://img.shields.io/badge/Lint-Ruff-E9EEFF?style=for-the-badge&logo=ruff&logoColor=4E5B7A&labelColor=D8E0F6" alt="Ruff"/>
  <img src="https://img.shields.io/badge/License-MIT-FFF0D9?style=for-the-badge&labelColor=F6E2BA&color=FFF0D9" alt="License"/>
</p>

<p><strong>A calm Python laboratory for experiments, algorithms, evidence, and better engineering habits.</strong></p>

</div>

## What velvet-python is

`velvet-python` is a learning repository, but not the kind that stops at "it runs."

The aim here is to make code understandable at multiple depths:

- the small implementation you can read in one sitting,
- the test that proves what it should do,
- the script that turns it into a repeatable workflow,
- the artifact that lets you compare one run against another later.

This is a place to learn Python by touching real moving parts: data validation, modeling baselines, experiment orchestration, CLI design, environment setup, debugging, and evidence trails.

The central rule is simple:

> Every useful claim should leave behind something another person can inspect.

## How the repository is organized

```text
velvet-python/
├─ src/
│  ├─ ai.py                 Small classifiers from first principles
│  ├─ ml_pipeline.py        Feature/model wrappers and evaluation helpers
│  ├─ data_utils.py         Validation, splitting, signatures, dataset profiles
│  ├─ pipeline.py           Reproducible run manifests and series summaries
│  ├─ evidence_ledger.py    Drift checks, evidence aggregation, review guidance
│  ├─ model_registry.py     Stable model catalog and metadata
│  └─ algorithm_guide.py    Recommendation logic for choosing approaches
├─ scripts/
│  ├─ run_experiments.py    Reproducible model sweeps and machine-readable reports
│  ├─ dataset_audit.py      Dataset quality checks and JSON evidence
│  └─ demos and utilities   Exploration, visuals, and environment helpers
├─ tests/
│  ├─ contract tests        Data, pipeline, CLI, and evidence behaviors
│  ├─ integration tests     End-to-end experiment and command checks
│  └─ benchmark tests       Lightweight performance sanity checks
├─ docs/
│  ├─ learning-path.md
│  ├─ experiment-ledger.md
│  ├─ practice-labs.md
│  └─ algorithm-package-playbook.md
├─ pyfiles/                 Sketchbook-style lessons, experiments, and walkthroughs
├─ environments/            Setup guides and environment automation
├─ CLI.py                   Typer command surface for learning by doing
├─ makefile                 Fast quality commands
└─ README.md
```

## The backbone that holds the repo together

### 1. Data is validated before it is trusted

`src/data_utils.py` does the unglamorous work that keeps later results honest:

- validates labeled text rows up front,
- rejects malformed records early,
- computes dataset signatures,
- builds lightweight quality profiles,
- supports deterministic train/test splitting.

That means the repo teaches two things at once:

1. how to write Python that works,
2. how to avoid fooling yourself with bad inputs.

### 2. Runs are treated like evidence, not vibes

`src/pipeline.py` builds a reproducible `ExperimentRun` record for every sweep.

Each run carries:

- the seed and holdout ratio,
- train/test sizes,
- accuracy and confusion matrix,
- dataset hash and quality profile,
- split-specific evidence,
- run duration,
- normalized parameters that can be serialized safely.

The result is a manifest you can compare, review, and rerun.

### 3. The ledger asks whether a result is sturdy

`src/evidence_ledger.py` is the review layer.

It validates manifest contracts, groups related runs, checks for drift, and turns repeated experiments into a small review packet:

- model-family summaries,
- dataset hash stability,
- spread across seeds,
- holdout consistency,
- practical recommendations when something starts to wobble.

This is where "I ran a model" becomes "I can explain whether this result is reliable."

## A good first path through the repo

If you are new here, this route works well:

1. Read `src/data_utils.py`
2. Read `src/pipeline.py`
3. Run `python scripts/run_experiments.py --epochs 2 --seed 42`
4. Open the generated JSON manifests in `artifacts/`
5. Compare the per-run `summary` with the grouped `series_summary`
6. Read `tests/test_pipeline_contracts.py` and `tests/test_evidence_ledger.py`

That path moves from input contracts, to runtime behavior, to stored evidence, to tests.

## Quick start

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Gate check

```bash
make install
make check
```

### Run an experiment sweep

```bash
python scripts/run_experiments.py --epochs 2 --seed 42 --summary-csv
```

That command writes:

- one JSON manifest per run,
- a machine-readable terminal report,
- a `summary.csv` for spreadsheet review,
- model-family rollups in the JSON report.

### Generate a dataset audit

```bash
python scripts/dataset_audit.py --source builtin --batch-size 2 --write-json artifacts/dataset.json
```

### Use the CLI entrypoint

```bash
python CLI.py pipeline --epochs 2 --seed 44 --output artifacts
```

## Interactive atlas

The repository also includes a GitHub Pages surface in `docs/`.

Planned Pages URL:

- [cazzy-aporbo.github.io/velvet-python](https://cazzy-aporbo.github.io/velvet-python/)

Rebuild the catalog data with:

```bash
make pages-catalog
```

Once GitHub Pages is enabled for the `docs/` folder on `main`, the atlas becomes a visual catalog of the Python files, learning tracks, and deeper study paths across the repo.

## What to look for in the outputs

### In a single manifest

Pay attention to:

- `dataset_hash`
- `dataset_profile`
- `split_profile`
- `confusion_matrix`
- `run_duration_seconds`
- `parameters`

Those fields usually explain more than the headline accuracy does.

### In the grouped summary

Look for:

- seed-to-seed spread,
- whether dataset hashes stayed stable,
- whether the holdout ratio moved,
- whether split label coverage became uneven.

If a result improves but the evidence around it gets noisier, that is worth slowing down for.

## Different ways to use this repository

| If you are... | Start here | Then move to |
|---|---|---|
| Learning Python fundamentals | `pyfiles/`, `src/data_utils.py` | `tests/test_data_pipeline_utils.py` |
| Practicing data engineering habits | `scripts/dataset_audit.py` | `src/evidence_ledger.py` |
| Practicing ML reasoning | `src/ml_pipeline.py`, `src/pipeline.py` | `scripts/run_experiments.py` |
| Preparing for interviews | `CLI.py`, `docs/experiment-ledger.md` | `tests/test_cli_pipeline_command.py` |
| Contributing to an open learning repo | `contributions.md` | `docs/practice-labs.md` |

## What the tests are doing for you

The test suite is part of the teaching surface, not an afterthought.

Useful places to read:

- `tests/test_data_pipeline_utils.py`
- `tests/test_pipeline_contracts.py`
- `tests/test_run_experiments_cli.py`
- `tests/test_evidence_ledger.py`
- `tests/test_cli_pipeline_command.py`
- `tests/test_data_utils_and_ingestion.py`

Run the focused set with:

```bash
pytest \
  tests/test_data_pipeline_utils.py \
  tests/test_pipeline_contracts.py \
  tests/test_run_experiments_cli.py \
  tests/test_evidence_ledger.py \
  tests/test_cli_pipeline_command.py
```

## What contributors should preserve

When you change something here, try to keep four qualities intact:

1. The code should still be readable by a person learning from it.
2. The behavior should still be pinned by tests.
3. The result should still leave behind an inspectable artifact.
4. The explanation should still sound like a person thinking carefully, not hiding behind abstraction.

## Supporting reading

- [docs/learning-path.md](docs/learning-path.md)
- [docs/experiment-ledger.md](docs/experiment-ledger.md)
- [docs/practice-labs.md](docs/practice-labs.md)
- [docs/algorithm-package-playbook.md](docs/algorithm-package-playbook.md)
- [contributions.md](contributions.md)

## License

MIT.

## Attribution

Built and maintained by [Cazzy Aporbo](https://github.com/Cazzy-Aporbo).
