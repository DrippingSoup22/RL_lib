# RL Lib

A small reinforcement-learning library that implements core algorithms from
first principles and validates them with Gymnasium experiments.

## Scope

| Family | Algorithms |
| --- | --- |
| Bandits | Epsilon-greedy and UCB; stationary and nonstationary |
| Monte Carlo | First/every visit prediction and epsilon-greedy control |
| Temporal difference | One-step and bounded-rollout TD prediction, SARSA, and Q-learning |
| Function approximation | Semi-gradient one-step and bounded-rollout TD, SARSA, and Q-learning |
| Policy gradients | REINFORCE with baseline, A2C, A3C, TRPO, and PPO |

Bandits, Monte Carlo, tabular temporal-difference, function approximation, the
two REINFORCE variants, and A2C are implemented currently. See
[`docs/roadmap.md`](docs/roadmap.md) for the planned order.

## Structure

```text
src/rl_lib/       reusable algorithms, policies, and trajectory data
experiments/      one Gymnasium runner per implemented family
tests/            small deterministic algorithm and environment tests
runs/             generated metrics, summaries, and recordings (ignored by Git)
```

## Setup and checks

```bash
python -m pip install -e ".[experiments,dev]"
make check
```

Make uses `python3` by default. To use a particular virtual environment without
repeating its path, create an ignored `local.mk` containing:

```make
PYTHON := /path/to/virtualenv/bin/python
```

Run the implemented experiments with:

```bash
make ba
make mc
make mc ARGS="--environment Taxi-v4"
make td
make td ARGS="--environment FrozenLake-v1"
make td ARGS="--preset quick --rollout 5"
make fa ARGS="-p quick"
make fa ARGS="-e MountainCar-v0 -p quick"
make fa ARGS="-p tuning --lr 0.001 --eps 0.1"
make fa ARGS="-p quick --rollout 5"
make pg ARGS="-p quick"
make pg ARGS="-e Acrobot-v1 -p tuning"
```

Run `make help` to see every experiment shortcut and accepted argument.

Monte Carlo runs use environment-specific `standard` budgets. Add
`--preset quick` for a short compatibility run; explicit episode counts still
override the selected preset.

Each run writes unsmoothed measurements to `metrics.csv`, complete configuration
to `metadata.json`, an automatically generated `summary.html`, and selected
evaluation recordings. Gymnasium outputs live under
`runs/<family>/<environment>/<timestamp>/`; the multi-condition bandit runner
uses `runs/bandits/<timestamp>/`. Standard runs record behavior at 25%, 50%,
75%, and 100%; quick runs record only 100%.
For non-bandit experiments, those recordings come from the seed with the best
final frozen evaluation. Summaries report every seed separately as well as the
across-seed aggregate.
