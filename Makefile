VENV ?= /home/daniele/.venvs/rl-lib
PYTHON ?= $(VENV)/bin/python

# Optional command-line values, for example:
#   make test TEST=tests/unit/test_q_learning.py
#   make experiment MODULE=experiments.runners.random_baseline ARGS="--environment frozen_lake"
TEST ?=
PYTEST_ARGS ?=
MODULE ?=
ARGS ?=
ENV ?=
EPISODES ?= 1
RENDER ?= auto
OUTPUT ?=
RUN_DIR ?=

.DEFAULT_GOAL := help

.PHONY: help setup test unit integration coverage lint format format-check check \
	baseline watch record experiment plot

help: ## Show the available commands.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install the project and development dependencies in the WSL venv.
	$(PYTHON) -m pip install -e ".[experiments,dev]"

test: ## Run tests; optionally set TEST=path and PYTEST_ARGS="...".
	$(PYTHON) -m pytest $(TEST) $(PYTEST_ARGS)

unit: ## Run the fast unit tests.
	$(PYTHON) -m pytest tests/unit $(PYTEST_ARGS)

integration: ## Run the Gymnasium integration tests.
	$(PYTHON) -m pytest tests/integration $(PYTEST_ARGS)

coverage: ## Run all tests and create a terminal coverage report.
	$(PYTHON) -m pytest --cov=rl_lib --cov-report=term-missing $(PYTEST_ARGS)

lint: ## Check the Python code with Ruff.
	$(PYTHON) -m ruff check .

format: ## Format the Python code with Ruff.
	$(PYTHON) -m ruff format .

format-check: ## Check formatting without changing files.
	$(PYTHON) -m ruff format --check .

check: lint format-check test ## Run every non-mutating quality check.

baseline: ## Run the random baseline; optionally set ENV=name and ARGS="...".
	$(PYTHON) -m experiments.runners.random_baseline $(if $(ENV),--environment $(ENV)) $(ARGS)

watch: ## Watch a random agent; optionally set ENV=name, EPISODES=count, RENDER=mode.
	$(PYTHON) -m experiments.runners.watch_environment $(if $(ENV),--environment $(ENV)) --episodes $(EPISODES) --render-mode $(RENDER) $(ARGS)

record: ## Record a random episode as a GIF; optionally set ENV=name and OUTPUT=path.
	$(PYTHON) -m experiments.runners.record_episode $(if $(ENV),--environment $(ENV)) $(if $(OUTPUT),--output $(OUTPUT)) $(ARGS)

experiment: ## Run a repository module, optionally with ARGS="...".
	@test -n "$(MODULE)" || { echo 'Usage: make experiment MODULE=experiments.<topic>.<runner> ARGS="..."'; exit 2; }
	$(PYTHON) -m $(MODULE) $(ARGS)

plot: ## Plot RUN_DIR=runs/<algorithm>/<run-id>, optionally with ARGS="...".
	@test -n "$(RUN_DIR)" || { echo 'Usage: make plot RUN_DIR=runs/<algorithm>/<run-id>'; exit 2; }
	$(PYTHON) -m experiments.runners.plot_results $(RUN_DIR) $(ARGS)
