-include local.mk

PYTHON ?= python3
TEST ?=
PYTEST_ARGS ?=
MODULE ?=
ARGS ?=
PRESET ?=
ENV ?=
ALGORITHM ?=
SEEDS ?=
SEED_BASE ?=
TRAIN ?=
EVAL ?=

COMMON_EXPERIMENT_ARGS = \
	$(if $(PRESET),--preset $(PRESET)) \
	$(if $(ENV),--environment $(ENV)) \
	$(if $(ALGORITHM),--algorithm $(ALGORITHM)) \
	$(if $(SEEDS),--seeds $(SEEDS)) \
	$(if $(SEED_BASE),--seed-base $(SEED_BASE)) \
	$(if $(TRAIN),--training-episodes $(TRAIN)) \
	$(if $(EVAL),--evaluation-episodes $(EVAL))

.DEFAULT_GOAL := help

.PHONY: help setup test coverage lint format format-check check experiment ba mc td fa pg

help: ## Show the available commands.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo 'Shared non-bandit variables:'
	@echo '  PRESET={quick,tuning,standard}  ENV=ID  ALGORITHM=NAME'
	@echo '  SEEDS=N  SEED_BASE=N  TRAIN=N  EVAL=N'
	@echo '  ARGS="..." passes family-specific options through unchanged.'
	@echo
	@echo 'Examples:'
	@echo '  make pg PRESET=standard ENV=CartPole-v1 ALGORITHM=a2c'
	@echo '  make fa PRESET=tuning ENV=Acrobot-v1 ALGORITHM=q_learning ARGS="--lr 0.001"'
	@echo '  make td PRESET=quick ENV=FrozenLake-v1 ALGORITHM=sarsa'
	@echo '  make fa PRESET=quick ENV=CartPole-v1 ALGORITHM=td_prediction'
	@echo '  make fa PRESET=quick ENV=Taxi-v4 ALGORITHM=q_learning'
	@echo 'Use make <family> ARGS="--help" for authoritative runner options.'

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

ba: ## Run the bandit experiment with optional ARGS="...".
	$(PYTHON) -m experiments.bandits.run $(ARGS)

mc: ## Run the Monte Carlo experiment with optional ARGS="...".
	$(PYTHON) -m experiments.monte_carlo.run $(COMMON_EXPERIMENT_ARGS) $(ARGS)

td: ## Run the temporal-difference experiment with optional ARGS="...".
	$(PYTHON) -m experiments.temporal_difference.run $(COMMON_EXPERIMENT_ARGS) $(ARGS)

fa: ## Run the function-approximation experiment with optional ARGS="...".
	$(PYTHON) -m experiments.function_approximation.run $(COMMON_EXPERIMENT_ARGS) $(ARGS)

pg: ## Run the policy-gradient experiment with optional ARGS="...".
	$(PYTHON) -m experiments.policy_gradient.run $(COMMON_EXPERIMENT_ARGS) $(ARGS)
