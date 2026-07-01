# Velvet Python Learning Path

This repository is arranged as an **engineering learning ladder**.
Every section links code, tests, and observable outputs so you can verify claims quickly.

## The path

### Stage 1: Core fluency

Goal: move from “I can run it” to “I can defend it.”

- Files: `src/data_utils.py`, `src/ml_pipeline.py`, `src/pipeline.py`
- Tasks:
  1. Run dataset load + validation manually.
  2. Run one train/test split and explain why the selected seed matters.
  3. Read one manifest and identify three quality signals.

Pass criteria:

- you can explain dataset shape violations,
- you can explain deterministic seed behavior,
- your local run is reproducible.

### Stage 2: Evidence-first experimentation

Goal: add rigor before tuning.

- Files: `scripts/run_experiments.py`, `tests/test_data_pipeline_utils.py`, `docs/experiment-ledger.md`
- Tasks:
  1. run experiments with `--epochs 3` and inspect summary payload,
  2. add `--summary-csv` and inspect output,
  3. compare one baseline to one altered split.

Pass criteria:

- manifests are generated and archived,
- you can compare two runs and identify what changed,
- you can explain why one metric change may still be a risk.

### Stage 3: Pipeline confidence and failure design

Goal: turn unknown behavior into explicit failure rules.

- Files: `tests/test_dataset_audit_cli.py`, `scripts/dataset_audit.py`, `src/data_utils.py`
- Tasks:
  1. add a malformed case to your local test set,
  2. verify failure type and message,
  3. record expected behavior in `docs/experiment-ledger.md`.

Pass criteria:

- broken inputs fail fast,
- failure mode is documented,
- recovery or mitigation path is clear.

### Stage 4: Role-based extension

Goal: connect repository patterns to real job skills.

- Files: `docs/practice-labs.md`, `README.md`, `contributions.md`
- Tasks:
  1. pick one role (data engineer / data scientist / ML engineer),
  2. write one targeted test for that concern,
  3. run `make check` and summarize results in plain language.

Pass criteria:

- you can communicate technical choices to a non-technical reader,
- you can justify tradeoffs with evidence,
- you can propose next-step hardening.

## Skills this path maps

- deterministic software engineering habits
- basic data validation and quality checks
- model baseline comparison with confidence in reproducibility
- pipeline thinking: where data enters, transforms, and leaves
- artifact-driven communication for engineering teams

## Practical transition questions

Try answering after each stage:

- Which assumption changed in this stage?
- What broke that assumption and what prevented the break?
- What metric now has stronger meaning than a single “accuracy” number?

If you can answer these without guessing, the stage is done.
