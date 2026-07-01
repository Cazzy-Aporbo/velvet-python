# Velvet Python Practice Labs

This guide is a structured path for building intuition from scripts to reliable systems.
Each lab has three parts:

1. **Action** – what to run or edit.
2. **Failure signal** – what wrong behavior to look for.
3. **Evidence** – what artifact proves learning.

## Lab 1: Baseline confidence (dataset contract)

**Action**

- Open `src/data_utils.py` and read `validate_dataset`.
- Run:

```bash
python scripts/dataset_audit.py --source builtin --write-json artifacts/audit.json
```

**Failure signal**

- audit script fails or returns empty profile.

**Evidence**

- JSON report contains `source`, `sample_count`, and `dataset_profile`.

## Lab 2: Split behavior and variance

**Action**

- Run two sweeps with different seeds:

```bash
python scripts/run_experiments.py --epochs 1 --seed 1 --test-ratio 0.30
python scripts/run_experiments.py --epochs 1 --seed 2 --test-ratio 0.30
```

**Failure signal**

- `accuracy` jumps wildly and cannot be explained by class balance.

**Evidence**

- Compare all manifest files and verify `dataset_hash` is identical.

## Lab 3: Add one safe model

**Action**

- Add a model in `CLI.py` and `scripts/run_experiments.py` registries.
- Add one test that checks the model appears in the selected model run list.

**Failure signal**

- script run breaks with `Unknown model` or no manifest produced.

**Evidence**

- test that filtered run (`--models`) outputs only that model.

## Lab 4: CSV ingestion realism

**Action**

- Create a minimal CSV with columns `text,label` and one intentionally bad row.
- Run:

```bash
python scripts/dataset_audit.py --source csv --csv path/to/file.csv --batch-size 3
```

**Failure signal**

- malformed data does not raise errors, or silently drops rows without trace.

**Evidence**

- explicit test in `tests/test_data_pipeline_utils.py` covering bad CSV inputs.

## Interview-style reflection prompts

Use these before every substantial repo change:

- What could this change silently break?
- What does a failed split or malformed row look like?
- What artifact proves this behavior did not regress after a refactor?

## Suggested progression cadence

- **Week 1:** Labs 1 and 2 + 1 PR-sized refactor
- **Week 2:** Labs 3 and 4 + review manifest diffs in PR notes
- **Week 4:** Write one short case study using one metric tradeoff observed from your runs

The intent is not to add files quickly. It is to reduce ambiguity while increasing skill depth.
