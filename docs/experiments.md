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
