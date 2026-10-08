# RL_lib: project rules

A compact, educational reinforcement-learning library: broad, clear algorithm
coverage without framework-level complexity.

## Scope

- Bandits: epsilon-greedy and UCB, stationary and nonstationary.
- Monte Carlo: first-visit and every-visit prediction, and epsilon-greedy
  control.
- Temporal difference: TD(0), SARSA and Q-learning.
- Function approximation: semi-gradient TD(0) prediction, SARSA and
  Q-learning.
- Policy gradients: REINFORCE with baseline, A2C, A3C, TRPO and PPO.

## Code

- Explain the mathematical update before the code.
- Reusable algorithms live in `src/rl_lib`, and Gymnasium experiments in
  `experiments`. `experiments/common.py` shares only mechanical runner
  helpers; environment semantics, training, evaluation and summaries stay in
  each family's runner.
- Keep runners CPU-first and sequential by default. Add vectorization or
  multiprocessing only for an algorithmic need or a measured bottleneck.
- Stop interaction on either termination or truncation. TD targets bootstrap
  after truncation, but not after true termination. Evaluation may still count
  a truncated episode as unsuccessful.
- When something fails, distinguish mathematical bugs, implementation bugs and
  hyperparameter issues.

## Tests

- Write one test per behaviour an update computes, and one test per class for
  its constructor checks. Add a Gymnasium interaction test only where no
  experiment-runner test already drives that agent on a real environment. Keep
  tests small and deterministic.
- Don't test per-call argument checks or experiment plumbing: command-line
  options, defaults, seed presets, run folders, dashboards, metrics or CSV
  files.
- After a change, run only the tests that cover it. Run the whole suite at the
  end of a piece of work.
- End-to-end validation is a multi-seed Gymnasium experiment. Run the relevant
  component tests before one starts.
- The user runs tuning and standard experiments. The assistant may run only
  smoke tests or the `quick` preset, to check experiment compatibility.

## Experiments

- Each experiment has one runner. It writes raw `metrics.csv` and
  `metadata.json`, an automatic `summary.html`, and only the selected
  behaviour recordings.
- Runs go under `runs/<environment>/<family>/<algorithm>/<timestamp>/`.
  Create only the exact path an experiment needs, and reuse existing parent
  directories. Synthetic experiments that cover several conditions in one run,
  such as bandits, may leave out the environment and algorithm levels.
- Print the output directory before work begins. For long runs, write the
  metadata before training, report algorithm, seed and checkpoint progress,
  and rewrite `metrics.csv` after each completed checkpoint, so that an
  interruption keeps completed measurements.
- Build `summary.html` deterministically from measured data: configuration,
  computed tables, figures and embedded recordings, but no hand-written
  conclusions or AI-generated interpretation. Show only key settings there;
  complete reproducibility details go in `metadata.json`.
- Standard experiments record frozen evaluation behaviour at 25%, 50%, 75% and
  100% of training. Quick compatibility runs may record only 100%, and tuning
  runs may omit recordings. Embed GIF or video recordings in the summary,
  never render normal training, and never update the agent during a
  recording.
- Start learning curves with a frozen 0% evaluation baseline when the metric
  is defined before training, but make no 0% behaviour recording.
- Smooth noisy learning curves for display by averaging each run before
  aggregation. Keep the raw CSV measurements unsmoothed, and record the
  smoothing window in the metadata.
- Use one primary environment and at most one challenge environment per
  experiment, unless another environment answers a specific question.
- Expose meaningful difficulty settings through concise CLI options or named
  presets, and store the resolved settings in `metadata.json`.
- Pair environment seeds across algorithm variants. For tabular agents, keep a
  generated MDP fixed for the whole training and evaluation run, so that state
  meanings don't change between episodes.

## Commands

RL_lib runs inside WSL (Ubuntu), where `make` and the environment that
`local.mk` points to are installed. It keeps that environment for now, so
don't create a `.venv`.

The Makefile uses `python3` by default. Machine-specific interpreter overrides
belong in the ignored `local.mk`, for example `PYTHON := /path/to/python`.
These commands use the interpreter the Makefile selects, and work from Zed:

```bash
make test
make lint
make format-check
make check
```
