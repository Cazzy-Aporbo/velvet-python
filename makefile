PYTHON := python
SRC_DIR := src
TEST_DIR := tests

.DEFAULT_GOAL := help
.PHONY: help install dev lint test test-fast clean check

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@$(PYTHON) -m pip install -U pip
	@pip install -r requirements.txt
	@pip install -e ".[dev]"

dev: install ## Install + run full check
	@$(MAKE) check

lint: ## Lint with ruff
	@ruff check $(SRC_DIR) $(TEST_DIR)

test: ## Run tests
	@pytest $(TEST_DIR) -v --tb=short

test-fast: ## Run tests (no output)
	@pytest $(TEST_DIR) -q

check: lint test ## Lint + test

clean: ## Remove caches and build artifacts
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build dist htmlcov .coverage 2>/dev/null || true
