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
two REINFORCE variants, A2C, A3C, and discrete-action PPO are implemented. See
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
make mc PRESET=quick ENV=Blackjack-v1 ALGORITHM=first_visit_control
make td PRESET=quick ENV=FrozenLake-v1 ALGORITHM=sarsa
make fa PRESET=quick ENV=CartPole-v1 ALGORITHM=td_prediction
make fa PRESET=quick ENV=Taxi-v4 ALGORITHM=q_learning
make fa PRESET=tuning ENV=Acrobot-v1 ALGORITHM=q_learning \
  ARGS="--lr 0.001 --eps 0.1"
make pg PRESET=standard ENV=CartPole-v1 ALGORITHM=a2c
make pg PRESET=quick ENV=CartPole-v1 ALGORITHM=ppo
```

Run `make help` to see every experiment shortcut and accepted argument.

Every non-bandit invocation runs one explicitly selected algorithm or variant.
`quick` is an artifact-free compatibility check, `tuning` writes a compact
configuration report, and `standard` performs the final multi-seed evaluation.
Explicit episode counts still override the selected preset.

Blackjack, CliffWalking, FrozenLake, and Taxi are accepted by every non-bandit
family. Compatibility is based on observation/action spaces rather than a
difficulty allowlist, so intentionally weak algorithm/environment combinations
remain available for educational experiments.

Quick runs write nothing. Tuning and standard Gymnasium outputs live under
`runs/<environment>/<family>/<algorithm>/<timestamp>/`. Only the branch needed
by the selected experiment is created; existing parent directories are reused.
Tuning writes raw measurements, metadata, and a compact HTML. Standard
additionally records the best seed at 25%, 50%, 75%, and 100%, saves that
selected policy as `best_model.pt` or `best_model.npz`, and produces the complete
multi-seed report. The multi-condition bandit runner keeps its existing
`runs/bandits/<timestamp>/` layout.
