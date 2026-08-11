# Experiments

Experiments are end-to-end Gymnasium validations, not reusable library code.
Each implemented family has one runner:

```bash
make experiment MODULE=experiments.bandits.run
make experiment MODULE=experiments.monte_carlo.run
make experiment MODULE=experiments.monte_carlo.run ARGS="--environment Taxi-v4"
make experiment MODULE=experiments.monte_carlo.run ARGS="--environment Taxi-v4 --preset quick"
make experiment MODULE=experiments.temporal_difference.run
make experiment MODULE=experiments.temporal_difference.run ARGS="--environment FrozenLake-v1"
make experiment MODULE=experiments.temporal_difference.run ARGS="--preset quick"
```

The temporal-difference runner evaluates TD(0) prediction, SARSA, and Q-learning.
It uses `CliffWalking-v1` as the primary environment and configurable generated
`FrozenLake-v1` maps as the challenge. Compared control algorithms share maps,
training seeds, and frozen greedy evaluation seeds.

The Monte Carlo runner accepts Blackjack and zero-based Gymnasium environments
with discrete observation and action spaces. Blackjack keeps its small custom
state encoder and value heatmaps; other environments use the generic tabular
path and a uniform-random prediction policy.

The `standard` preset is the multi-seed experiment and chooses budgets suited to
each environment's episode length. It records all four evaluation checkpoints.
The `quick` preset is only a short pipeline check and records the final policy.
Explicit episode-count and seed options override either preset.

Summaries also follow environment semantics. Blackjack reports value heatmaps
and win/draw/loss rates. Taxi reports value distributions, success, episode
length, truncation, and illegal actions. Unknown discrete environments use only
generic return, length, truncation, coverage, and value-distribution metrics.

Runs use paired seeds and normally compare one primary environment with one
challenge environment. Difficulty is selected through CLI options or named
presets and the resolved configuration is saved in `metadata.json`.

Evaluation behavior is frozen and sampled at 25%, 50%, 75%, and 100% of
training. Renderable environments produce an embedded GIF or video per
checkpoint; non-visual environments use a learning curve. Normal training is
never rendered. Outputs remain under the ignored `runs/` directory.

Learning curves also include a frozen 0% baseline when their metric is defined
before training. The 0% baseline is measured but not recorded as a GIF.

`metrics.csv` and `metadata.json` are the raw results. `summary.html` is a
deterministic dashboard built from them with configuration tables, computed
metrics, figures, and embedded recordings. It contains no manually authored
conclusions or AI-generated interpretation.

The dashboard shows only the settings needed to read the result; complete
configuration remains in `metadata.json`. Noisy curves may use a documented
moving average calculated independently for every run. Raw CSV measurements are
never smoothed.

Blackjack is a special case because one hand often ends after a single action.
Its checkpoint GIFs concatenate five fixed-seed evaluation hands; environments
with longer trajectories record one complete episode per checkpoint.

For generated tabular environments, create one map per seed and keep it fixed
through training and evaluation. Compared variants must receive the same maps.
