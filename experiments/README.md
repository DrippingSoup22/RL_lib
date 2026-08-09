# Experiments

Experiments are end-to-end Gymnasium validations, not reusable library code.
Each implemented family has one runner:

```bash
make experiment MODULE=experiments.bandits.run
make experiment MODULE=experiments.monte_carlo.run
```

Runs use several seeds and write only `metrics.csv`, `metadata.json`, and
`report.md` under the ignored `runs/` directory.
