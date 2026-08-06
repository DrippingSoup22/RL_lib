# RL Lib

A small reinforcement-learning library for implementing algorithms from first
principles, comparing them in Gymnasium environments, and reusing them in future
projects.

## Repository layout

```text
src/rl_lib/
├── algorithms/       # Learning algorithms, grouped by family
│   ├── actor_critic/
│   ├── policy_gradient/
│   ├── tabular/
│   └── value_based/
├── data/             # Trajectories, replay buffers, and batch structures
├── evaluation/       # Evaluation loops and metrics
└── models/           # Reusable function approximators

experiments/          # Environment configs and executable comparisons
results/              # Curated summaries, plots, and conclusions
runs/                 # Generated raw metrics and checkpoints (ignored)
docs/                 # Concise architecture and workflow documentation

tests/
├── unit/             # Fast tests for individual library components
└── integration/      # Algorithms interacting with environments
```

`src/rl_lib` is the installable library. Experiments may import it, but the
library never imports project-level experiments or results.

## Setup

Create and activate a virtual environment, then install the project in editable
mode with its experiment and development tools:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[experiments,dev]"
```

Run the test suite with:

```bash
pytest
```

Before any algorithms are implemented, validate the configured environments and
the result pipeline with a random-policy baseline:

```bash
python -m experiments.random_baseline
python -m experiments.plot_results runs/random_baseline/<run-id>
```

The first command prints the exact run directory. The second creates a report
under `results/random_baseline/<run-id>` by default.

## Design rules

- Keep algorithms independent of a particular Gymnasium environment.
- Keep transition and rollout storage in `data`, separate from learning logic.
- Add abstractions only after a concrete algorithm needs them.
- Keep reusable model definitions in `models`; trained weights belong in
  ignored run directories.
- Keep generated output in `runs`; commit only selected summaries and figures
  to `results`.
- Mirror library modules under `tests/unit`; reserve `tests/integration` for
  end-to-end environment interactions.

See [`docs/architecture.md`](docs/architecture.md) for the boundaries and
[`docs/experiments.md`](docs/experiments.md) for the experiment contract.
