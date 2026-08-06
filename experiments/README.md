# Experiments

This directory connects library algorithms to configured Gymnasium environments.
It is repository tooling, not part of the installable `rl_lib` package.

Environment definitions live in `environments/*.json`. They contain only
Gymnasium construction arguments and baseline evaluation settings. Algorithm
hyperparameters will live separately when each algorithm is introduced.

To check all configured environments and establish a random-policy baseline:

```bash
python -m experiments.random_baseline
```

The command writes a timestamped directory under `runs/random_baseline`. Raw run
directories are intentionally ignored by Git.

To aggregate a run and produce graphs:

```bash
python -m experiments.plot_results runs/random_baseline/<run-id>
```

Every future experiment runner should emit the same `episodes.csv` columns
described in `docs/experiments.md`. This lets the plotting tool compare future
algorithms without knowing how they are implemented.

