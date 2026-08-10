# RL Lib

A small reinforcement-learning library that implements core algorithms from
first principles and validates them with Gymnasium experiments.

## Scope

| Family | Algorithms |
| --- | --- |
| Bandits | Epsilon-greedy and UCB; stationary and nonstationary |
| Monte Carlo | First/every visit prediction and epsilon-greedy control |
| Temporal difference | TD(0), SARSA, and Q-learning |
| Function approximation | Semi-gradient SARSA and Q-learning |
| Policy gradients | REINFORCE with baseline, A2C, A3C, TRPO, and PPO |

Only bandits and Monte Carlo are implemented currently. See
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

Run the implemented experiments with:

```bash
make experiment MODULE=experiments.bandits.run
make experiment MODULE=experiments.monte_carlo.run
make experiment MODULE=experiments.monte_carlo.run ARGS="--environment Taxi-v4"
```

Monte Carlo runs use environment-specific `standard` budgets. Add
`--preset quick` for a short compatibility run; explicit episode counts still
override the selected preset.

Each run writes raw metrics and metadata, an automatically generated visual
`summary.html`, and selected evaluation recordings under `runs/`. Standard runs
record 25%, 50%, 75%, and 100%; quick runs record only 100%.
