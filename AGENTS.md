# Project guidance

This is a compact educational reinforcement-learning library. Its goal is broad,
clear algorithm coverage without framework-level complexity.

## Scope

- Bandits: epsilon-greedy and UCB, stationary and nonstationary.
- Monte Carlo: first/every visit prediction and epsilon-greedy control.
- Temporal difference: TD(0), SARSA, and Q-learning.
- Function approximation: semi-gradient TD(0) prediction, SARSA, and Q-learning.
- Policy gradients: REINFORCE with baseline, A2C, A3C, TRPO, and PPO.

Do not implement algorithms outside the user's current request.

## Approach

- Explain the mathematical update before code.
- Keep reusable algorithms in `src/rl_lib` and Gymnasium experiments in
  `experiments`.
- Share only mechanical runner helpers in `experiments/common.py`; keep
  environment semantics, training, evaluation, and summaries in each family's
  runner.
- Prefer direct implementations over speculative abstractions.
- Keep tests few, small, and deterministic: one test per behaviour an update
  computes, one test per class for its constructor checks, and a Gymnasium
  interaction test only where no experiment-runner test already drives that
  agent on a real environment. Do not test per-call argument checks or
  experiment plumbing: command-line options, defaults, seed presets, run
  folders, dashboards, metrics, or CSV files.
- After a change, run only the tests that cover it; run the whole suite at the
  end of a piece of work.
- Treat a multi-seed Gymnasium experiment as the end-to-end validation.
- Run the relevant component tests before starting an end-to-end experiment.
- The user runs tuning and standard experiments. The assistant may run only
  smoke tests or the `quick` preset to verify experiment compatibility.
- Keep each experiment to one runner. Write raw `metrics.csv` and
  `metadata.json`, an automatic `summary.html`, and only the selected behavior
  recordings.
- Store Gymnasium runs under
  `runs/<environment>/<family>/<algorithm>/<timestamp>/`. Create only the exact
  branch required by an experiment and reuse any existing parent directories.
  Synthetic experiments that cover several conditions in one run, such as
  bandits, may omit the environment and algorithm directories.
- Print the output directory before work begins. For long runs, write metadata
  before training, report algorithm/seed/checkpoint progress, and rewrite
  `metrics.csv` after each completed checkpoint so interruption preserves
  completed measurements.
- Build `summary.html` deterministically from measured data. Include
  configuration, computed tables, figures, and embedded recordings, but no
  manually authored conclusions or AI-generated interpretation.
- Show only key settings in `summary.html`; keep complete reproducibility details
  in `metadata.json`.
- Standard experiments record frozen evaluation behavior at 25%, 50%, 75%,
  and 100% of training. Quick compatibility runs may record only 100%. Embed
  GIF or video recordings in the summary, never render normal training, and
  never update the agent during a recording. Tuning runs may omit recordings.
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
- Keep runners CPU-first and sequential by default. Add vectorization or
  multiprocessing only when it answers an algorithmic need or a measured
  bottleneck justifies the extra complexity.
- Stop interaction on either termination or truncation. TD targets bootstrap
  after truncation but not after true termination; evaluation may still count a
  truncated episode as unsuccessful.
- Distinguish mathematical bugs, implementation bugs, and hyperparameter issues.

## Commands

The Makefile uses `python3` by default. Machine-specific interpreter overrides
belong in the ignored `local.mk`, for example `PYTHON := /path/to/python`.

```bash
make test
make lint
make format-check
make check
```

These commands use the interpreter selected by the Makefile and work from Zed.
