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
make td ARGS="--preset quick --rollout 5"
make fa
make fa ARGS="-e MountainCar-v0"
make fa ARGS="-p quick"
make fa ARGS="-p tuning --lr 0.001 --eps 0.1"
make fa ARGS="-p quick --rollout 5"
make pg ARGS="-p quick"
make pg ARGS="-e Acrobot-v1 -p tuning"
```

`make help` lists every shortcut and all accepted runner arguments. The generic
`make experiment MODULE=experiments.<family>.run` form remains available.

The temporal-difference runner evaluates rollout TD prediction, SARSA, and
Q-learning. `--rollout` selects the maximum rollout length and defaults to one,
retaining the original one-step behavior. Like A2C, each update builds returns
backward from one endpoint bootstrap and uses every transition in the bounded
rollout. `CliffWalking-v1` is the primary environment and configurable generated
`FrozenLake-v1` maps are the challenge. Compared control algorithms share maps,
training seeds, and frozen greedy evaluation seeds.

The Monte Carlo runner accepts Blackjack and zero-based Gymnasium environments
with discrete observation and action spaces. Blackjack keeps its small custom
state encoder and value heatmaps; other environments use the generic tabular
path and a uniform-random prediction policy.

The function-approximation runner compares semi-gradient rollout SARSA with
rollout Q-learning on continuous observations and discrete actions. It also
accepts `--rollout` as the maximum rollout length, defaulting to the former
one-step behavior. Semi-gradient rollout TD prediction is covered by
deterministic component tests rather than this control experiment.
The runner uses `Acrobot-v1` as the primary environment and `MountainCar-v0` as
the challenge. Observations are statically rescaled to `[-1, 1]`; compared
agents share environment seeds and initial network parameters. Summaries report
frozen greedy success and an environment-specific progress measurement: maximum
position for MountainCar and maximum tip height for Acrobot.

The policy-gradient runner compares REINFORCE, REINFORCE with a learned
state-value baseline, and A2C on `CartPole-v1`, with `Acrobot-v1` as the
challenge. Observations are flattened to `float32` without rescaling.
REINFORCE updates after each complete trajectory; A2C updates during an episode
from bounded n-step rollouts, bootstrapping at ordinary rollout boundaries and
after truncation but not true termination. `--rollout` controls the maximum
rollout length and defaults to five transitions. `--entropy` controls A2C's
optional entropy bonus and defaults to zero for the unregularized comparison.
All three algorithms run by default; `--algorithm a2c` selects only A2C for
focused tuning, while `--algorithms reinforce_with_baseline a2c` selects an
ordered subset.
Evaluation samples the frozen categorical policy: CartPole succeeds by reaching
its time limit, while Acrobot succeeds by truly terminating at its goal.

Policy-gradient actor and critic learning rates follow independent cosine
schedules stepped once per training episode. By default, a linear warmup covers
the first ten percent of training and starts at one tenth of each peak learning
rate; `--warmup` sets its episode count or disables it with zero, while
`--warmup-start` sets the initial factor. `--lr` and `--lr-min` configure the
actor; `--critic-lr` and `--critic-lr-min` configure the baseline and A2C
critics. Each minimum defaults to one percent of its peak. The refined defaults
use actor and critic peaks of `0.003` and `0.01`. The runner supports SGD, Adam,
and AdamW through `-o`; the default is AdamW with `--wd 0.0001` applied to both
networks. Optional `--diag` training diagnostics write one compact row per
completed training episode to `training_diagnostics.csv`, including learning
rates, losses, policy entropy, and A2C's number of rollout updates.

The `standard` preset is the multi-seed experiment and chooses budgets suited to
each environment's episode length. It records all four evaluation checkpoints.
The `quick` preset is only a short pipeline check and records the final policy.
Function approximation also has a one-seed `tuning` preset with a moderate
budget and no recordings. Policy gradients use the same three preset names.
Explicit episode-count and seed options override any preset.

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

For every non-bandit algorithm, recordings show one coherent run: the seed with
the highest final mean frozen-evaluation return, with success rate as the main
tiebreaker. All requested checkpoints come from that seed. Summaries still show
the across-seed aggregate and a separate final row for every seed, and metadata
records the selected visual seed.

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
