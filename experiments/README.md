# Experiments

Experiments are end-to-end Gymnasium validations, not reusable library code.
Each implemented family has one runner:

```bash
make ba
make mc
make mc ARGS="--environment Taxi-v4"
make mc ARGS="--environment Taxi-v4 --preset quick"
make td
make td ARGS="--environment FrozenLake-v1"
make td ARGS="--preset quick"
make fa
make fa ARGS="-e MountainCar-v0"
make fa ARGS="-p quick"
make fa ARGS="-p tuning --lr 0.001 --eps 0.1"
```

`make help` lists every shortcut and all accepted runner arguments. The generic
`make experiment MODULE=experiments.<family>.run` form remains available.

The temporal-difference runner evaluates TD(0) prediction, SARSA, and Q-learning.
It uses `CliffWalking-v1` as the primary environment and configurable generated
`FrozenLake-v1` maps as the challenge. Compared control algorithms share maps,
training seeds, and frozen greedy evaluation seeds.

The Monte Carlo runner accepts Blackjack and zero-based Gymnasium environments
with discrete observation and action spaces. Blackjack keeps its small custom
state encoder and value heatmaps; other environments use the generic tabular
path and a uniform-random prediction policy.

The function-approximation runner compares semi-gradient SARSA with Q-learning
on continuous observations and discrete actions. Semi-gradient TD(0) prediction
is covered by deterministic component tests rather than this control experiment.
The runner uses `Acrobot-v1` as the primary environment and `MountainCar-v0` as
the challenge. Observations are statically rescaled to `[-1, 1]`; compared
agents share environment seeds and initial network parameters. Summaries report
frozen greedy success and an environment-specific progress measurement: maximum
position for MountainCar and maximum tip height for Acrobot.

The `standard` preset is the multi-seed experiment and chooses budgets suited to
each environment's episode length. It records all four evaluation checkpoints.
The `quick` preset is only a short pipeline check and records the final policy.
Function approximation also has a one-seed `tuning` preset with a moderate
budget and no recordings. Explicit episode-count and seed options override any
preset.

The function-approximation runner keeps its full option names but also provides
short forms for repeated tuning work: `-e` for environment, `-p` for preset,
`-o` for optimizer, `-n` for seeds, plus `--train`, `--eval`, `--lr`, `--gamma`,
`--eps`, `--sarsa-eps-final`, `--hidden`, and `-r` for recording mode. SARSA
epsilon decays linearly to `--sarsa-eps-final`, which defaults to `0.01`, while
Q-learning keeps `--eps` constant because its greedy target is already
off-policy. Adam is the default optimizer. Its learning rate follows cosine
decay from `--lr` to `--lr-min`, which defaults to one percent of the initial
rate. At each checkpoint, `--val` fixed-seed episodes select the best model seen
so far; reported evaluation uses different held-out seeds. Recording mode is
`none`, `final`, or `checkpoints`. Optional `--diag` training diagnostics write
one compact row per training episode to `training_diagnostics.csv`, including
the epsilon and learning rate used for that episode. Validation and held-out
measurements remain distinguishable in `metrics.csv`.

Summaries also follow environment semantics. Blackjack reports value heatmaps
and win/draw/loss rates. Taxi reports value distributions, success, episode
length, truncation, and illegal actions. Unknown discrete environments use only
generic return, length, truncation, coverage, and value-distribution metrics.

Runs use paired seeds and normally compare one primary environment with one
challenge environment. Difficulty is selected through CLI options or named
presets and the resolved configuration is saved in `metadata.json`.

Evaluation behavior is frozen and sampled at 25%, 50%, 75%, and 100% of
training. Function approximation reports the best policy selected on separate
fixed-seed validation episodes at or before each checkpoint. Renderable
environments produce an embedded GIF or video per checkpoint; non-visual
environments use a learning curve. Normal training is never rendered. Gymnasium outputs use
`runs/<family>/<environment>/<timestamp>/`. The bandit runner covers both of its
synthetic conditions in one `runs/bandits/<timestamp>/` directory.

Learning curves also include a frozen 0% baseline when their metric is defined
before training. The 0% baseline is measured but not recorded as a GIF.

`metrics.csv` contains unsmoothed measurements and `metadata.json` contains the
resolved reproducibility settings. `summary.html` is a deterministic dashboard
built from measured data with configuration tables, computed metrics, figures,
and embedded recordings. It contains no manually authored conclusions or
AI-generated interpretation.

The dashboard shows only the settings needed to read the result; complete
configuration remains in `metadata.json`. Noisy curves may use a documented
moving average calculated independently for every run. Raw CSV measurements are
never smoothed.

Blackjack is a special case because one hand often ends after a single action.
Its checkpoint GIFs concatenate five fixed-seed evaluation hands; environments
with longer trajectories record one complete episode per checkpoint.

For generated tabular environments, create one map per seed and keep it fixed
through training and evaluation. Compared variants must receive the same maps.

The runners share only small filesystem, checkpoint, CSV/JSON, and frame-label
helpers from `experiments/common.py`. Environment construction, learning,
evaluation metrics, and summary contents stay in the family runner. Long runs
print their output path immediately, report algorithm/seed/checkpoint progress,
write metadata before training, and preserve metrics after each completed
checkpoint.

Experiments run sequentially on CPU by default. Vectorized environments or
multiprocessing should be introduced only for an algorithmic reason or after a
measured bottleneck makes the added complexity worthwhile.
