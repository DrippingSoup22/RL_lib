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

The epsilon-greedy experiment compares three initial action values against three
epsilon values in a 3-by-3 convergence grid:

```bash
make experiment MODULE=experiments.bandits.exp_EpsilonGreedyBandits
```

It uses deterministic rewards by default. The same experiment can use stationary
stochastic rewards without duplicating the runner:

```bash
make experiment MODULE=experiments.bandits.exp_EpsilonGreedyBandits ARGS="--reward-std 1"
```

It writes per-step data, a summary, metadata, and `epsilon_greedy_grid.png`
under `runs/bandits/epsilon_greedy/<run-id>/`.

Future episodic Gymnasium runners should emit the same `episodes.csv` columns
described in `docs/experiments.md`. Non-episodic experiments such as bandits use
topic-specific per-step data instead of inventing artificial episodes.
