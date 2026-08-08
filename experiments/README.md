# Experiments

This directory connects library algorithms to configured Gymnasium environments.
It is repository tooling, not part of the installable `rl_lib` package.

Generic command-line tools live in `runners/`. Algorithm-specific directories,
such as `bandits/` or `monte_carlo/`, should be added only when their first
experiment is implemented. The experiment directories intentionally omit
`__init__.py`: they are repository-local Python namespaces, not installed
packages.

Environment definitions live in `environments/*.json`. They contain only
Gymnasium construction arguments and baseline evaluation settings. Algorithm
hyperparameters will live separately when each algorithm is introduced.

To check all configured environments and establish a random-policy baseline:

```bash
make baseline
make baseline ENV=cart_pole
```

The command writes a timestamped directory under `runs/random_baseline`. Raw run
directories are intentionally ignored by Git.

To watch a random agent interact with an environment:

```bash
make watch
make watch ENV=cart_pole EPISODES=5
```

`frozen_lake` is the default because it is the smallest deterministic task.
Toy-text environments render directly in the terminal by default, avoiding
Pygame window problems under WSLg. Classic-control environments use a graphical
window. A mode can be selected explicitly with `RENDER=ansi` or `RENDER=human`.
Graphical rendering requires a working WSLg display and the experiment extras
installed by `make setup`.

For a visual result that does not depend on WSLg, record an animated GIF and
open it from Windows or Zed:

```bash
make record ENV=cart_pole
make record ENV=frozen_lake OUTPUT=runs/previews/lake.gif
```

Recordings use a random policy for now. A trained algorithm should record its
evaluation policy through the same RGB-frame approach, making before/after
behavior easy to compare.

## Environment progression

The environments introduce one main source of difficulty at a time:

| Step | Environment | Main idea introduced |
| --- | --- | --- |
| 1 | `frozen_lake` | Discrete states/actions, deterministic transitions |
| 2 | `cliff_walking` | Larger deterministic control problem |
| 3 | `frozen_lake_slippery` | Stochastic transitions |
| 4 | `cart_pole` | Continuous observations |
| 5 | `mountain_car` | Continuous observations and sparse progress |
| 6 | `pendulum` | Continuous observations and actions |

These are all stationary, sequential, episodic tasks. Episodic means that an
interaction has an end; it does not mean that actions within the episode are
independent. MuJoCo can come later for higher-dimensional continuous control,
but it is deliberately not a dependency yet. Its simulated physics is not
automatically stochastic, so randomness should be added and measured explicitly
if an experiment needs it.

To aggregate a run and produce graphs:

```bash
make plot RUN_DIR=runs/random_baseline/<run-id>
```

Any repository experiment module can be invoked through the generic target. For
example, this is equivalent to `make baseline ENV=frozen_lake`:

```bash
make experiment MODULE=experiments.runners.random_baseline ARGS="--environment frozen_lake"
```

Bandit metric definitions and the experiments that use them are documented in
[`bandits/README.md`](bandits/README.md).

The epsilon-greedy experiment compares neutral, mildly optimistic, and strongly
optimistic initial action values. For each initial value, it contrasts greedy
action selection (`epsilon=0`) with low (`epsilon=0.01`) and conventional
(`epsilon=0.1`) persistent exploration:

```bash
make experiment MODULE=experiments.bandits.epsilon_stationary
```

It uses the standard stationary stochastic test bed by default: true action
values and reward noise both have unit standard deviation. Deterministic rewards
remain useful for inspecting sample-average updates in isolation:

```bash
make experiment MODULE=experiments.bandits.epsilon_stationary ARGS="--reward-std 0"
```

The figure mirrors the UCB experiment's outcome measures—mean reward,
optimal-action rate, and value-estimation error—while using one column per
initial value. Shaded regions are approximate 95% confidence intervals across
runs. The run writes per-step data, a summary, metadata, and
`epsilon_greedy_grid.png` under `runs/bandits/epsilon_greedy/<run-id>/`.

The nonstationary epsilon-greedy experiment gives every true action value an
independent Gaussian random walk. It compares the sample-average update against
a constant step size while keeping the environment and agent seeds paired:

```bash
make experiment MODULE=experiments.bandits.epsilon_nonstationary
```

The defaults compare epsilon values `0`, `0.01`, and `0.1` against the
sample-average update and constant step sizes `0.01` and `0.1`. They use drift
standard deviation `0.01`, 3,000 interactions, and 200 independent runs. The
3-by-3 figure uses one epsilon per column and one metric per row, with shared
row scales and consistent update-rule colors. It writes per-step data, a
final-step summary, metadata, and
`nonstationary_epsilon_greedy.png` under
`runs/bandits/nonstationary_epsilon_greedy/<run-id>/`. Epsilon values and step
sizes can be changed without editing the runner, for example:

```bash
make experiment MODULE=experiments.bandits.epsilon_nonstationary ARGS="--epsilons 0.01 0.05 0.1 --constant-step-sizes 0.02 0.1"
```

The figure shows pseudo-regret, optimal-action rate, and estimation MSE using a
100-step rolling mean by default. Smoothing is applied independently to every
run before confidence intervals are calculated; CSV measurements remain raw.
Use `--smoothing-window 1` for an unsmoothed figure.

The UCB experiment compares a forced-initialization greedy baseline (`c=0`)
with moderate (`c=1`) and stronger (`c=2`) confidence bonuses:

```bash
make experiment MODULE=experiments.bandits.ucb_stationary
```

It uses stationary stochastic rewards by default because deterministic rewards
make every estimate exact after UCB's first visit to each arm. The initial value
is fixed at zero: with forced first visits and sample-average updates, changing
it cannot affect the trajectory. Custom exploration constants can be supplied
without changing the runner:

```bash
make experiment MODULE=experiments.bandits.ucb_stationary ARGS="--exploration-constants 0 0.5 1 2"
```

The run writes per-step data, a summary, metadata, and `ucb_c_sweep.png` under
`runs/bandits/ucb_greedy/<run-id>/`. It uses the same bandit evaluator and
confidence-interval convention as the epsilon-greedy experiment.

The nonstationary UCB experiment applies the same `c`-by-update-rule grid used
for nonstationary epsilon-greedy:

```bash
make experiment MODULE=experiments.bandits.ucb_nonstationary
```

Its columns use cumulative-count UCB constants `0`, `1`, and `2`; each panel
compares sample-average estimates with constant step sizes `0.01` and `0.1`.
Rows show pseudo-regret, optimal-action rate, and estimation MSE. Defaults are
3,000 interactions, 200 runs, and a 100-step rolling visualization; raw CSV
measurements remain unsmoothed.

This is intentionally standard UCB evaluated under drift. Constant step sizes
help observed value estimates track changes, but the UCB confidence counts are
not forgotten. It is a baseline for a later discounted or sliding-window UCB,
not a fully nonstationary confidence rule.

Future episodic Gymnasium runners should emit the same `episodes.csv` columns
described in `docs/experiments.md`. Non-episodic experiments such as bandits use
topic-specific per-step data instead of inventing artificial episodes.
