-include local.mk

PYTHON ?= python3
TEST ?=
PYTEST_ARGS ?=
MODULE ?=
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help setup test coverage lint format format-check check experiment ba mc td fa pg

help: ## Show the available commands.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo 'Experiment arguments (pass as ARGS="..."):'
	@echo '  ba  --runs N --steps N --smoothing-window N --seed N'
	@echo '  mc  --environment ID --preset {quick,standard}'
	@echo '      --prediction-episodes N --training-episodes N'
	@echo '      --evaluation-episodes N --epsilon FLOAT --seeds N'
	@echo '  td  --environment {CliffWalking-v1,FrozenLake-v1}'
	@echo '      --preset {quick,standard} --prediction-episodes N'
	@echo '      --training-episodes N --evaluation-episodes N --seeds N'
	@echo '      --learning-rate FLOAT --discount FLOAT --epsilon FLOAT'
	@echo '      --rollout/--n-steps N'
	@echo '      --map-size N --safe-probability FLOAT --non-slippery'
	@echo '      --max-episode-steps N'
	@echo '  fa  -e/--env/--environment {Acrobot-v1,MountainCar-v0}'
	@echo '      -p/--preset {quick,tuning,standard}'
	@echo '      -o/--opt/--optimizer {sgd,adam}'
	@echo '      --train/--training-episodes N --eval/--evaluation-episodes N'
	@echo '      -n/--seeds N --lr/--learning-rate FLOAT --lr-min FLOAT'
	@echo '      --val/--validation-episodes N'
	@echo '      --gamma/--discount FLOAT --eps/--epsilon FLOAT'
	@echo '      --rollout/--n-steps N'
	@echo '      --sarsa-eps-final/--sarsa-final-epsilon FLOAT'
	@echo '      --hidden/--hidden-sizes N [N ...]'
	@echo '      -r/--record/--recordings {none,final,checkpoints}'
	@echo '      --diag/--diagnostics'
	@echo '  pg  -e/--env/--environment {CartPole-v1,Acrobot-v1}'
	@echo '      -p/--preset {quick,tuning,standard}'
	@echo '      --algorithm/--algorithms {reinforce,reinforce_with_baseline,a2c} [...]'
	@echo '      --train/--training-episodes N --eval/--evaluation-episodes N'
	@echo '      -n/--seeds N --lr/--actor-lr FLOAT --lr-min FLOAT'
	@echo '      --critic-lr FLOAT --critic-lr-min FLOAT --wd/--weight-decay FLOAT'
	@echo '      -o/--opt/--optimizer {sgd,adam,adamw} --gamma/--discount FLOAT'
	@echo '      --warmup/--warmup-episodes N --warmup-start FLOAT'
	@echo '      --rollout/--a2c-rollout-steps N --entropy FLOAT'
	@echo '      --hidden/--hidden-sizes N [N ...]'
	@echo '      -r/--record/--recordings {none,final,checkpoints}'
	@echo '      --diag/--diagnostics'

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
	$(PYTHON) -m experiments.monte_carlo.run $(ARGS)

td: ## Run the temporal-difference experiment with optional ARGS="...".
	$(PYTHON) -m experiments.temporal_difference.run $(ARGS)

fa: ## Run the function-approximation experiment with optional ARGS="...".
	$(PYTHON) -m experiments.function_approximation.run $(ARGS)

pg: ## Run the policy-gradient experiment with optional ARGS="...".
	$(PYTHON) -m experiments.policy_gradient.run $(ARGS)
