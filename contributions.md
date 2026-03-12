# Contributing

Fork, clone, branch, test, PR.

```bash
git clone https://github.com/YOUR-USERNAME/velvet-python.git
cd velvet-python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Before Submitting

```bash
make check   # runs ruff + pytest
```

All tests must pass. If you add code to `src/`, add a corresponding test in `tests/`.

## Commit Messages

```
type: brief description

fix: handle empty input in classify_text
feat: add YAML config loader to data_utils
test: benchmark prediction throughput
```

## Code Style

- Ruff for linting, type hints where practical
- No `====` separators in comments
- Docstrings on public functions, skip the obvious ones
- If a comment restates the code, delete it

## Reporting Issues

Open an issue with: Python version, OS, what you expected, what happened, minimal reproduction.
