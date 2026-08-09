VENV ?= /home/daniele/.venvs/rl-lib
PYTHON ?= $(VENV)/bin/python
TEST ?=
PYTEST_ARGS ?=
MODULE ?=
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help setup test coverage lint format format-check check experiment

help: ## Show the available commands.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install library, experiment, and development dependencies.
	$(PYTHON) -m pip install -e ".[experiments,dev]"

test: ## Run tests; optionally set TEST=path and PYTEST_ARGS="...".
	$(PYTHON) -m pytest $(TEST) $(PYTEST_ARGS)

coverage: ## Run tests with a terminal coverage report.
	$(PYTHON) -m pytest --cov=rl_lib --cov-report=term-missing $(PYTEST_ARGS)

lint: ## Check Python code with Ruff.
	$(PYTHON) -m ruff check .

format: ## Format Python code with Ruff.
	$(PYTHON) -m ruff format .

format-check: ## Check formatting without changing files.
	$(PYTHON) -m ruff format --check .

check: lint format-check test ## Run all quality checks.

experiment: ## Run MODULE=experiments.<family>.run with optional ARGS="...".
	@test -n "$(MODULE)" || { echo 'Usage: make experiment MODULE=experiments.<family>.run ARGS="..."'; exit 2; }
	$(PYTHON) -m $(MODULE) $(ARGS)
