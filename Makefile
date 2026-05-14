# Deterministic Horizon — developer Makefile
# Tested on macOS, Linux, and Git-Bash on Windows.

PY ?= python
PIP ?= $(PY) -m pip
EXAMPLES = examples

.PHONY: help install dev demo test lint format clean paper-figures paper-tables

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## pip install -e .
	$(PIP) install -e .

dev: ## pip install -e ".[dev,all]"
	$(PIP) install -e ".[dev,all]"
	pre-commit install || true

demo: ## Run the offline demo (no API keys)
	$(PY) $(EXAMPLES)/demo.py

test: ## Run unit tests (skips slow/api)
	$(PY) -m pytest -q -m "not slow and not api"

test-all: ## Run all tests including slow
	$(PY) -m pytest -q

lint: ## Lint with ruff
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests

format: ## Auto-format
	$(PY) -m ruff check --fix src tests
	$(PY) -m black src tests

clean: ## Remove build artefacts
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

paper-figures: ## Regenerate every figure in the paper from cached results
	$(PY) -c "from deterministic_horizon.analysis import generate_figures; \
              generate_figures('results/sample/synthetic_results.json', 'assets', fmt='png')"

paper-tables: ## Regenerate every table in the paper from cached results
	$(PY) -c "from deterministic_horizon.analysis import generate_tables; \
              generate_tables('results/sample/synthetic_results.json', 'analysis')"
