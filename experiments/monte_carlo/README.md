# Monte Carlo experiments

This section will keep three questions separate:

1. **Prediction:** estimate the value of a fixed policy.
2. **Policy improvement:** construct a better policy from action-value estimates.
3. **Control:** find a good policy by alternating approximate prediction and
   improvement through generalized policy iteration.

The initial control method will be on-policy and epsilon-soft. Environments will
use their normal reset distribution; exploration will come from stochastic action
selection throughout each episode rather than exploring starts.

First-visit and every-visit estimates will be compared in the prediction
experiment. Off-policy Monte Carlo and importance sampling are outside the initial
scope.

The control comparison can be run with:

```bash
make experiment MODULE=experiments.monte_carlo.control
```

Each run writes the repository-standard `episodes.csv`, Monte Carlo-specific
`training.csv`, final learned tables, metadata, and a human-readable `report.md`
under `runs/monte_carlo/control/<run-id>/`. The episode data and report keep the
epsilon-soft training performance separate from evaluation with a frozen greedy
policy. The number of evaluation episodes can be changed with
`--evaluation-episodes`.

The prediction comparison can be run with:

```bash
make experiment MODULE=experiments.monte_carlo.prediction
```

It learns a policy with first-visit MC control, freezes an epsilon-soft copy,
and evaluates that same policy and the same episode stream with first-visit and
every-visit prediction. Each run writes `episodes.csv`, `prediction.csv`, final
estimates, metadata, and `report.md` under
`runs/monte_carlo/prediction/<run-id>/`.
