# Project guidance

This is a compact educational reinforcement-learning library. Its goal is broad,
clear algorithm coverage without framework-level complexity.

## Scope

- Bandits: epsilon-greedy and UCB, stationary and nonstationary.
- Monte Carlo: first/every visit prediction and epsilon-greedy control.
- Temporal difference: TD(0), SARSA, and Q-learning.
- Function approximation: semi-gradient SARSA and Q-learning.
- Policy gradients: REINFORCE with baseline, A2C, A3C, TRPO, and PPO.

Do not implement algorithms outside the user's current request.

## Approach

- Explain the mathematical update before code.
- Keep reusable algorithms in `src/rl_lib` and Gymnasium experiments in
  `experiments`.
- Prefer direct implementations over speculative abstractions.
- Add only small deterministic tests for update correctness, validation, and
  environment compatibility. Do not unit-test report or CSV plumbing.
- Treat a multi-seed Gymnasium experiment as the end-to-end validation.
- Keep each experiment to one runner and write only `metrics.csv`,
  `metadata.json`, and `report.md`.
- Distinguish mathematical bugs, implementation bugs, and hyperparameter issues.

## Commands

The WSL virtual environment is `/home/daniele/.venvs/rl-lib`.

```bash
make test
make lint
make format-check
make check
```

These commands use the virtual environment explicitly and work from Zed.
