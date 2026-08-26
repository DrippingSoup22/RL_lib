# Experiments

Experiments are end-to-end Gymnasium validations, not reusable library code.
Bandits retain their synthetic multi-condition runner. Every other invocation
runs one explicitly selected algorithm or variant.

## Modes

`quick` is an artifact-free compatibility check. It uses a tiny deterministic
budget, trains and evaluates the selected algorithm, and writes nothing under
`runs/`.

`tuning` is a one-seed hyperparameter run by default. It writes raw metrics,
complete metadata, one learning figure, and a compact `summary.html`. It does
not record behavior or save a model.

`standard` is the final multi-seed experiment. It evaluates frozen policies at
0%, 25%, 50%, 75%, and 100% of training, plots every seed and the aggregate,
reports aggregate and per-seed results, records the selected best seed at the
nonzero checkpoints, and saves the same selected policy as `best_model.pt` or
`best_model.npz`.

A future comparison mode will cover paired multi-algorithm experiments. It is
not currently exposed by the CLI.

## Running experiments

The Makefile exposes shared variables and leaves family-specific settings in
`ARGS`:

```bash
make mc PRESET=quick ENV=Blackjack-v1 ALGORITHM=first_visit_control
make mc PRESET=standard ENV=Taxi-v4 ALGORITHM=every_visit_control
make td PRESET=quick ENV=FrozenLake-v1 ALGORITHM=sarsa
make fa PRESET=tuning ENV=Acrobot-v1 ALGORITHM=q_learning \
  ARGS="--lr 0.001 --eps 0.1"
make fa PRESET=quick ENV=Taxi-v4 ALGORITHM=q_learning
make pg PRESET=standard ENV=CartPole-v1 ALGORITHM=a2c
make pg PRESET=quick ENV=Acrobot-v1 ALGORITHM=a3c ARGS="--workers 2"
make pg PRESET=quick ENV=CartPole-v1 ALGORITHM=ppo \
  ARGS="--batch-episodes 4 --update-epochs 4 --minibatch-size 64"
```

Use `make <family> ARGS="--help"` for the authoritative family-specific CLI.

Monte Carlo variants are `first_visit_prediction`, `every_visit_prediction`,
`first_visit_control`, and `every_visit_control`. Temporal-difference variants
are `td_prediction`, `sarsa`, and `q_learning`. Function approximation provides
`sarsa` and `q_learning`. Policy gradients provide `reinforce`,
`reinforce_with_baseline`, `a2c`, `a3c`, and discrete-action `ppo`.

## Outputs

Persisted Gymnasium runs use:

```text
runs/<environment>/<family>/<algorithm>/<timestamp>/
```

Only the exact branch required by an experiment is created, and existing parent
directories are reused. Deleting any portion of `runs/` therefore does not
affect later experiments. The experiment mode is stored in `metadata.json`.
Custom Gymnasium IDs are converted to a safe directory component while their
exact value remains in metadata. Bandits keep `runs/bandits/<timestamp>/`
because one invocation covers multiple synthetic conditions.

Raw `metrics.csv` values remain unsmoothed. Complete reproducibility settings
live in `metadata.json`; summaries display only the settings needed to read the
result. Training never renders, evaluation never updates the agent, and
interaction stops on either termination or truncation. Bootstrapped algorithms
retain the final-observation bootstrap after truncation but not true
termination.

Standard seed trials use paired environment seeds. Generated tabular maps stay
fixed for a complete seed trial. The chosen recording/model seed is the highest
final frozen-evaluation result using the family-specific tie breakers.

All four Gymnasium Toy Text environments—Blackjack, CliffWalking, FrozenLake,
and Taxi—can run in Monte Carlo, temporal-difference, function-approximation,
and policy-gradient experiments. No family blocks an environment because it is
unlikely to solve it: compatibility is determined by observation and action
spaces. Tabular runners accept finite `Discrete` observations or tuples of
`Discrete` values. Neural runners flatten fixed-size observations, which gives
Toy Text environments one-hot inputs. Every runner currently requires a
discrete action space; Pendulum and continuous MountainCar therefore require a
future continuous-action implementation.

Arbitrary registered environments are accepted when those same space
requirements are met. Gymnasium's `module:Env-v0` syntax can import personal
registrations. CliffWalking receives a 200-step episode limit in families where
an untrained policy would otherwise be able to wander indefinitely.
