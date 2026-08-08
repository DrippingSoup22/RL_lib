# Experiments

Environment definitions in `experiments/environments` are deliberately separate from algorithms. This allows the same algorithm to be run against multiple environments and multiple algorithms to be compared under one environment.

Each run writes a `metadata.json` file and an `episodes.csv` file. The episode schema is:

| Column | Meaning |
| --- | --- |
| `run_id` | Unique execution identifier |
| `algorithm` | Stable algorithm name |
| `environment` | Repository environment name |
| `gym_id` | Exact Gymnasium environment ID |
| `seed` | Random seed for the run |
| `phase` | Usually `training` or `evaluation` |
| `episode` | Zero-based episode number |
| `return` | Sum of rewards in the episode |
| `length` | Number of environment steps |
| `terminated` | Whether the task reached a terminal state |
| `truncated` | Whether an external limit ended the episode |

Future episodic algorithm runners should preserve this schema. Algorithm-specific data, such as losses or epsilon values, can go in a separate `training.csv` rather than making the common episode data inconsistent. Non-episodic problems such as bandits should record their natural interaction steps in a topic-specific CSV instead of inventing artificial episodes.

Comparisons should use several seeds, the same evaluation budget, and the exact environment configuration recorded by the run.

## Bandit measurements

Bandit experiments share the interaction and measurement code in
`experiments/bandits/evaluation.py`. Every parameter condition is evaluated with
the same environment and agent seeds so comparisons are paired.

The complete definitions, interpretation, timing, output columns, and current
figure consumers are documented in
[`experiments/bandits/README.md`](../experiments/bandits/README.md).

At each interaction the evaluator records reward, instantaneous pseudo-regret,
optimal-action selection, fraction of actions tried, and estimate MSE. The true
values and optimal action are captured before the environment step. This timing
also works for a nonstationary environment that changes its action values during
`step`.

Bandit runs write aggregate per-step measurements to `convergence.csv` and
final-step aggregates to `summary.csv`. Both files identify the algorithm and
condition parameters, and include means and standard deviations across runs.
Figures show mean curves with approximate 95% confidence intervals; the CSV
measurements themselves are not smoothed.

For nonstationary bandits, estimate MSE is measured against the true values that
generated the current reward, before the environment applies its random-walk
drift. The next interaction is evaluated against the updated true values.
