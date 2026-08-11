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
  environment compatibility. Do not unit-test dashboard or CSV plumbing.
- Treat a multi-seed Gymnasium experiment as the end-to-end validation.
- Run the relevant component tests before starting an end-to-end experiment.
- Keep each experiment to one runner. Write raw `metrics.csv` and
  `metadata.json`, an automatic `summary.html`, and only the selected behavior
  recordings.
- Build `summary.html` deterministically from measured data. Include
  configuration, computed tables, figures, and embedded recordings, but no
  manually authored conclusions or AI-generated interpretation.
- Show only key settings in `summary.html`; keep complete reproducibility details
  in `metadata.json`.
- Standard experiments record frozen evaluation behavior at 25%, 50%, 75%,
  and 100% of training. Quick compatibility runs may record only 100%. Embed
  GIF or video recordings in the summary, never render normal training, and
  never update the agent during a recording.
- Start learning curves with a frozen 0% evaluation baseline when the metric is
  defined before training. Do not create a 0% behavior recording.
- Smooth noisy learning curves for display by averaging each run before
  aggregation. Keep raw CSV measurements unsmoothed and record the smoothing
  window in metadata.
- Use one primary and at most one challenge environment per experiment unless
  another environment answers a specific question.
- Expose meaningful difficulty settings through concise CLI options or named
  presets, and store the resolved settings in `metadata.json`.
- Pair environment seeds across algorithm variants. For tabular agents, keep a
  generated MDP fixed for the whole training/evaluation run so state meanings do
  not change between episodes.
- Stop interaction on either termination or truncation. TD targets bootstrap
  after truncation but not after true termination; evaluation may still count a
  truncated episode as unsuccessful.
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
