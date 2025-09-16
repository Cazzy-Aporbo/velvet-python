# Velvet Python - Development Automation
# =======================================
# Author: Cazzy Aporbo, MS
# Created: January 2025
#
# I built this Makefile because I got tired of typing the same commands
# over and over. After the 100th time running "black . && isort . && ruff check .",
# I realized I needed automation. This file captures my entire development
# workflow - the commands I actually use every day.
#
# Usage: make [command]
# Run 'make help' to see all available commands

# Configuration
# -------------
# I set these as variables so I can easily change them later if needed
PYTHON := python
PIP := pip
PROJECT_NAME := velvet-python
PYTHON_VERSION := 3.11

# Directories - where everything lives in my project
SRC_DIR := velvet_python
TEST_DIR := tests
DOCS_DIR := docs
BENCHMARK_DIR := benchmarks

# Colors for output - because plain terminal output is boring
# I spent way too long picking these colors to match the pastel theme
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
RESET := \033[0m

# Default target - what happens when you just type 'make'
.DEFAULT_GOAL := help

# PHONY targets - these aren't actual files, just command names
.PHONY: help install dev-install format lint test clean docs run benchmark all

# Help Command
# ------------
# I always forget what commands are available, so I made this help system.
# It automatically extracts the comments from each command.
help: ## Show this help message (default)
	@echo "$(CYAN)╔══════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║$(RESET)  $(MAGENTA)Velvet Python - Development Commands$(RESET)                   $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)  Author: Cazzy Aporbo, MS                               $(CYAN)║$(RESET)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Example: make install$(RESET)"
	@echo ""

# Installation Commands
# ---------------------
# I separated install commands because sometimes I just want production deps,
# and sometimes I need everything for development.

install: ## Install production dependencies only
	@echo "$(YELLOW)Installing production dependencies...$(RESET)"
	@echo "This installs only what's needed to run the project"
	@$(PIP) install -U pip setuptools wheel
	@$(PIP) install -e .
	@echo "$(GREEN)✓ Production installation complete!$(RESET)"

dev-install: ## Install all development dependencies (recommended for contributors)
	@echo "$(YELLOW)Installing development dependencies...$(RESET)"
	@echo "This might take a while - I'm installing a lot of tools"
	@$(PIP) install -U pip setuptools wheel
	@$(PIP) install -r requirements-dev.txt
	@$(PIP) install -e .
	@echo ""
	@echo "$(CYAN)Installing pre-commit hooks...$(RESET)"
	@pre-commit install
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"
	@echo ""
	@echo "$(MAGENTA)Next steps:$(RESET)"
	@echo "  1. Run 'make test' to verify everything works"
	@echo "  2. Run 'make format' to format your code"
	@echo "  3. Run 'make help' anytime you forget a command"

# Code Quality Commands
# ---------------------
# These are the commands I run constantly during development.
# I probably run 'make format' 50 times a day.

format: ## Format code with black and isort (I run this before every commit)
	@echo "$(YELLOW)Formatting code...$(RESET)"
	@echo "Running black - my favorite formatter"
	@black $(SRC_DIR) tests/ benchmarks/ examples/ 2>/dev/null || black .
	@echo "Running isort - keeps imports organized"
	@isort $(SRC_DIR) tests/ benchmarks/ examples/ 2>/dev/null || isort .
	@echo "$(GREEN)✓ Code formatted successfully!$(RESET)"
	@echo "Your code now follows consistent style guidelines"

lint: ## Run linting checks with ruff and mypy (catches bugs before they happen)
	@echo "$(YELLOW)Running linters...$(RESET)"
	@echo ""
	@echo "$(CYAN)1/3: Running ruff (fast Python linter)...$(RESET)"
	@ruff check . || (echo "$(RED)✗ Ruff found issues$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Ruff check passed$(RESET)"
	@echo ""
	@echo "$(CYAN)2/3: Running mypy (type checking)...$(RESET)"
	@mypy $(SRC_DIR) --ignore-missing-imports || (echo "$(YELLOW)⚠ Type checking found issues$(RESET)")
	@echo ""
	@echo "$(CYAN)3/3: Running pylint (additional checks)...$(RESET)"
	@pylint $(SRC_DIR) --exit-zero || true
	@echo ""
	@echo "$(GREEN)✓ Linting complete!$(RESET)"

format-check: ## Check if code needs formatting (useful in CI)
	@echo "$(YELLOW)Checking code format...$(RESET)"
	@black --check $(SRC_DIR) tests/ benchmarks/ examples/ 2>/dev/null || \
		(echo "$(RED)✗ Code needs formatting. Run 'make format'$(RESET)" && exit 1)
	@isort --check-only $(SRC_DIR) tests/ benchmarks/ examples/ 2>/dev/null || \
		(echo "$(RED)✗ Imports need sorting. Run 'make format'$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Code format is correct!$(RESET)"

# Testing Commands
# ----------------
# I'm obsessed with testing. These commands run different test configurations
# depending on what I'm working on.

test: ## Run tests with coverage (my most-used command after format)
	@echo "$(YELLOW)Running tests with coverage...$(RESET)"
	@echo "This shows which code paths are tested"
	@pytest tests/ -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "$(GREEN)✓ Tests complete!$(RESET)"
	@echo "$(CYAN)Coverage report generated at: htmlcov/index.html$(RESET)"

test-fast: ## Run tests without coverage (when I need quick feedback)
	@echo "$(YELLOW)Running tests (fast mode)...$(RESET)"
	@pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete!$(RESET)"

test-failed: ## Re-run only failed tests (saves time during debugging)
	@echo "$(YELLOW)Re-running failed tests...$(RESET)"
	@pytest tests/ -v --lf
	@echo "$(GREEN)✓ Failed tests re-run complete!$(RESET)"

test-module: ## Test a specific module (usage: make test-module MODULE=09-concurrency)
	@if [ -z "$(MODULE)" ]; then \
		echo "$(RED)Error: MODULE not specified$(RESET)"; \
		echo "Usage: make test-module MODULE=09-concurrency"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Testing module: $(MODULE)...$(RESET)"
	@pytest $(MODULE)/tests/ -v
	@echo "$(GREEN)✓ Module tests complete!$(RESET)"

# Benchmark Commands
# ------------------
# Performance matters. I run these to make sure my "optimizations"
# actually make things faster, not slower.

benchmark: ## Run all benchmarks (warning: this takes a while)
	@echo "$(YELLOW)Running benchmarks...$(RESET)"
	@echo "Go grab a coffee - this will take a few minutes"
	@for dir in */benchmarks; do \
		if [ -d "$$dir" ]; then \
			echo "$(CYAN)Benchmarking $$(dirname $$dir)...$(RESET)"; \
			python -m pytest $$dir -v --benchmark-only; \
		fi \
	done
	@echo "$(GREEN)✓ All benchmarks complete!$(RESET)"

benchmark-module: ## Benchmark a specific module (usage: make benchmark-module MODULE=09-concurrency)
	@if [ -z "$(MODULE)" ]; then \
		echo "$(RED)Error: MODULE not specified$(RESET)"; \
		echo "Usage: make benchmark-module MODULE=09-concurrency"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Benchmarking module: $(MODULE)...$(RESET)"
	@python -m pytest $(MODULE)/benchmarks/ -v --benchmark-only
	@echo "$(GREEN)✓ Module benchmark complete!$(RESET)"

# Documentation Commands
# ----------------------
# I believe good documentation is as important as good code.
# These commands help me maintain both.

docs: ## Build documentation (generates the website)
	@echo "$(YELLOW)Building documentation...$(RESET)"
	@mkdocs build
	@echo "$(GREEN)✓ Documentation built!$(RESET)"
	@echo "$(CYAN)Output: site/index.html$(RESET)"

docs-serve: ## Serve documentation locally (hot-reload enabled)
	@echo "$(YELLOW)Starting documentation server...$(RESET)"
	@echo "$(CYAN)Documentation will be available at: http://localhost:8000$(RESET)"
	@echo "$(WHITE)Press Ctrl+C to stop$(RESET)"
	@mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(YELLOW)Deploying documentation to GitHub Pages...$(RESET)"
	@mkdocs gh-deploy --force
	@echo "$(GREEN)✓ Documentation deployed!$(RESET)"

# Cleaning Commands
# -----------------
# Sometimes you just need a fresh start. These commands clean up
# all the cruft that accumulates during development.

clean: ## Clean build artifacts and cache files
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	@echo "Removing Python cache files"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ htmlcov/ site/ 2>/dev/null || true
	@echo "$(GREEN)✓ Clean complete!$(RESET)"
	@echo "Your project is squeaky clean"

clean-deep: ## Deep clean including virtual environments (use with caution!)
	@echo "$(RED)Warning: This will remove virtual environments!$(RESET)"
	@echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
	@sleep 3
	@make clean
	@echo "$(YELLOW)Removing virtual environments...$(RESET)"
	@rm -rf .venv/ venv/ env/ .env/ 2>/dev/null || true
	@echo "$(GREEN)✓ Deep clean complete!$(RESET)"
	@echo "You'll need to run 'make dev-install' again"

# Running Commands
# ----------------
# Quick ways to run various parts of the project.

run: ## Run the CLI (shortcut for 'python -m velvet_python.cli')
	@echo "$(CYAN)Starting Velvet Python CLI...$(RESET)"
	@$(PYTHON) -m velvet_python.cli

run-module: ## Run a module's interactive app (usage: make run-module MODULE=09-concurrency)
	@if [ -z "$(MODULE)" ]; then \
		echo "$(RED)Error: MODULE not specified$(RESET)"; \
		echo "Usage: make run-module MODULE=09-concurrency"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Starting interactive app for: $(MODULE)...$(RESET)"
	@cd $(MODULE) && streamlit run app.py

# Quality Check Commands
# ----------------------
# These are my "pre-commit" checks. I run these before pushing code.

check: format-check lint test ## Run all quality checks (format, lint, test)
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)✓ All quality checks passed!$(RESET)"
	@echo "$(GREEN)════════════════════════════════════════$(RESET)"
	@echo "Your code is ready to commit"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(YELLOW)Running pre-commit hooks...$(RESET)"
	@pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks passed!$(RESET)"

# Development Workflow Commands
# -----------------------------
# These combine multiple commands for common workflows.

dev: dev-install format lint test ## Complete development setup and verification
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"
	@echo "$(GREEN)════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(MAGENTA)You're all set! Here are some useful commands:$(RESET)"
	@echo "  make format  - Format your code"
	@echo "  make test    - Run tests"
	@echo "  make docs-serve - View documentation"
	@echo "  make run     - Start the CLI"

ci: format-check lint test ## Run CI pipeline locally (mimics GitHub Actions)
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)✓ CI pipeline passed locally!$(RESET)"
	@echo "$(GREEN)════════════════════════════════════════$(RESET)"
	@echo "Your code should pass GitHub Actions"

# Module Management Commands
# --------------------------
# I use these to quickly create new modules with the right structure.

new-module: ## Create a new module structure (usage: make new-module MODULE=24-new-topic)
	@if [ -z "$(MODULE)" ]; then \
		echo "$(RED)Error: MODULE not specified$(RESET)"; \
		echo "Usage: make new-module MODULE=24-new-topic"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Creating new module: $(MODULE)...$(RESET)"
	@mkdir -p $(MODULE)/{src,tests,benchmarks,examples,notebooks,docs}
	@touch $(MODULE)/README.md
	@touch $(MODULE)/requirements.txt
	@touch $(MODULE)/app.py
	@touch $(MODULE)/src/__init__.py
	@touch $(MODULE)/tests/test_core.py
	@echo "$(GREEN)✓ Module structure created!$(RESET)"
	@echo "$(CYAN)Next steps:$(RESET)"
	@echo "  1. Edit $(MODULE)/README.md"
	@echo "  2. Add dependencies to $(MODULE)/requirements.txt"
	@echo "  3. Start coding in $(MODULE)/src/"

# Statistics Commands
# -------------------
# I like to track my progress and see how the project is growing.

stats: ## Show project statistics (lines of code, test coverage, etc.)
	@echo "$(CYAN)════════════════════════════════════════$(RESET)"
	@echo "$(CYAN)Project Statistics$(RESET)"
	@echo "$(CYAN)════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(YELLOW)Lines of Code:$(RESET)"
	@find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*" | xargs wc -l | tail -1
	@echo ""
	@echo "$(YELLOW)Number of Python files:$(RESET)"
	@find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*" | wc -l
	@echo ""
	@echo "$(YELLOW)Number of tests:$(RESET)"
	@find . -name "test_*.py" -not -path "./.venv/*" | wc -l
	@echo ""
	@echo "$(YELLOW)Number of modules:$(RESET)"
	@ls -d [0-9][0-9]-* 2>/dev/null | wc -l
	@echo ""
	@echo "$(CYAN)════════════════════════════════════════$(RESET)"

# All Command
# -----------
# When I want to be absolutely sure everything is perfect.

all: clean dev-install format lint test docs ## Complete rebuild and verification
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)✓ Complete build and verification successful!$(RESET)"
	@echo "$(GREEN)════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(MAGENTA)Everything is built, tested, and documented.$(RESET)"
	@echo "$(MAGENTA)Your project is in perfect condition.$(RESET)"

# Special Targets
# ---------------
# This tells make which targets aren't actual files

.PHONY: help install dev-install format lint test clean docs run benchmark \
        all check pre-commit dev ci new-module stats format-check test-fast \
        test-failed test-module benchmark-module docs-serve docs-deploy \
        clean-deep run-module

# End of Makefile
# ---------------
# Author: Cazzy Aporbo, MS
# Remember: Just type 'make' to see all available commands!
