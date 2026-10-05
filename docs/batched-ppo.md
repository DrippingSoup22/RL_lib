# Batched PPO

PPO is extended so that it can serve many agents acting in many parallel
environments at once, on the CPU or the GPU. The first user is the Centipede
project, where eight independent agents each act in thousands of simulated
worlds on every step. The change is deliberately narrow: one PPO, made batched
and tensor-based, and nothing else.

## Why

PPO currently handles one observation at a time and converts its data through
NumPy:

- `sample_action` takes one observation and returns Python numbers, so acting
  in `W` environments takes `W` separate calls, each waiting for its result.
- `update` converts every input with NumPy, so tensors on a GPU cannot enter it.
- `generalized_advantage_estimates` walks one trajectory that ends in one way.
- `ObservationNormalizer` updates its statistics one observation at a time.

The mathematics is unaffected. The networks, the policies, the clipped
objective, the entropy bonus, the separate actor and critic optimizers, and
gradient clipping already work on batches and on any device.

## Changes

| Part | Change |
| --- | --- |
| `algorithms/policy_gradient/ppo.py` | `sample_action`, `select_action`, and `state_value` take a batch of observations `(B, observation_size)` and return tensors on the networks' device. `update` takes tensors on any device, shuffles with a PyTorch generator, and returns one summary of the update instead of one record per minibatch. |
| `algorithms/policy_gradient/advantages.py` | GAE over a block of `T` steps in `W` environments, as PyTorch tensors. Each step says whether its episode ended there, and whether by termination; the value of the state after each step is given, so time limits and the end of a collection window bootstrap correctly. Replaces the single-trajectory NumPy version. |
| `normalization.py` | The same normaliser and modes, in PyTorch: running statistics are updated from a whole batch at once, on any device. |
| `experiments/policy_gradient/` | The PPO runner and the observation wrapper pass a batch of one. `Pendulum-v1` is added to the known environments. |

Checks of hyperparameters and shapes stay. Checks that every value is finite are
no longer made on every call, because on a GPU each such check makes the CPU
wait for the GPU; the experiment runner already checks observations as they
arrive from Gymnasium.

## Reorganisation

Made where the change touches the code anyway.

Folders and files, so that each name says what it holds (tests follow the same
layout):

| Before | After | Why |
| --- | --- | --- |
| `models/` | `networks/` | In reinforcement learning a "model" usually means a model of the environment; these are neural networks. |
| `data/` | `trajectories/` | It holds episodes and rollouts. |
| `optimizers/` | `optimization/` | It holds gradient clipping and shared optimizer state, not optimizers. |
| `optimizers/gradients.py` | `optimization/gradient_clipping.py` | It only clips gradients. |
| `optimizers/shared.py` | `optimization/shared_optimizer.py` | Says what is shared. |
| `data/normalization.py` | `normalization.py` | It processes observations rather than storing trajectories. |
| `data/policy_gradient.py` | merged into `ppo.py` | The records are used only by PPO. |

Names in the code that PPO uses:

| Before | After | Why |
| --- | --- | --- |
| `actor_model`, `critic_model` | `actor_network`, `critic_network` | Matches `networks/`; renamed in all policy-gradient algorithms, which share them with the experiment runner. |
| `policy.model` | `policy.network` | The network behind a policy. |
| `DiscretePolicyNetwork` | `CategoricalPolicyNetwork` | Pairs with `CategoricalPolicy`, as `GaussianPolicyNetwork` pairs with `SquashedGaussianPolicy`. |
| `_transform_action` | `_to_environment_action` | Turns the Gaussian sample into the bounded action. |
| `actions` in `PPO.update` | `policy_actions` | For continuous PPO these are the stored latent samples, not the actions the environment received. |
| `action` in the PPO samples | `environment_action` | The action sent to the environment, next to `latent_action`. |

Names for the rewritten parts:

| Before | After | Why |
| --- | --- | --- |
| PPO `seed`, `rng` | `shuffle_seed`, `shuffle_generator` | They only set the order of minibatches. |
| `PPOUpdateResult` | `PPOUpdateSummary` | It summarises a whole update. |
| `entropy` (a mean) in the PPO loss | `mean_entropy` | Distinguishes it from the per-sample values. |
| `…_tensor`, `…_array` suffixes | dropped | NumPy conversions disappear. |
| GAE `delta` | `td_error` | The one-step temporal-difference error. |
| Normaliser `count`, `clip` | `observation_count`, `clip_limit` | Each holds a number. |
| `normalize(update=…)` | `normalize(update_statistics=…)` | Says what is updated. |

Kept on purpose: `select_action` and `sample_action`, a convention across the
library (choose an action to execute, or sample one with what learning needs);
`state_value`, the standard name for V(s); the standard PPO terms
`clip_ratio`, `entropy_coefficient`, `approximate_kl`, `clip_fraction`,
`advantages`, and `return_targets`; and `low` and `high`, Gymnasium's names for
bounds.

Also: the test suite was cut from 352 cases to 145 under the testing rule in
`AGENTS.md`, keeping one test per computed behaviour and one constructor test
per class, and dropping per-call argument checks, experiment plumbing, and
repeats of the same interaction for every environment.
`tests/algorithms/policy_gradient/test_ppo.py` (52 of the 145) and
`tests/test_normalization.py` are rewritten for the batched interface in the
same way. The README and the roadmap record continuous PPO as implemented.

## Not included

The other algorithms and families, TRPO, a vectorised Gymnasium runner, rollout
storage (each user of PPO keeps its own), and evaluating several agents'
networks in one operation.

## Validation

- Unit tests: GAE matches hand-computed values, including a block in which
  environments end by termination, by time limit, and by the window ending, at
  different steps; the normaliser gives the same statistics whether a batch is
  added whole or in parts; a batch gives the same log-probabilities and values
  as its rows one at a time; an update runs on CUDA when a GPU is available.
- End to end: continuous PPO on `Pendulum-v1`, the first standard validation of
  continuous PPO, and a repeat of the discrete `CartPole-v1` standard run to
  confirm that nothing regressed.

## Status

| Step | Status |
| --- | --- |
| Folder, file, and PPO name changes | Done |
| Batched GAE | Done: tests rewritten, PPO runner adapted |
| Batched normaliser | Done: tests rewritten, experiment wrapper adapted |
| Batched PPO | Done: one `PPOActionSample` record, update summarised once |
| Experiments adapted, tests rewritten, README and roadmap | Done; Pendulum has no success criterion in the reports, so it is judged by return |
| CartPole standard run | Passed (`runs/CartPole-v1/policy_gradient/ppo/20261005T151407374209Z`) |
| Pendulum tuning run | Learns (`runs/Pendulum-v1/policy_gradient/ppo/20261005T152847439240Z`) |
| Pendulum standard run | Passed (`runs/Pendulum-v1/policy_gradient/ppo/20261005T172417206109Z`) |

**CartPole result (2026-10-05).** The batched PPO repeated the standard run of
2026-08-26 with the same learning settings and paired seeds (932104 to 932106).
All three seeds reached a frozen-policy return of 500 with 100% success, as
before. The untrained policies' returns at the 0% checkpoint are identical in
both runs (20.3, 18.3, 22.1), so the starting networks and evaluation episodes
match; the learning curves then differ only slightly (seed 932106 at 25%: 370
before, 295 now, 100% from 50% on in both), as expected from the new shuffling
generator and a newer PyTorch. This validates categorical PPO, GAE, and the
time-limit bootstrap end to end; it does not exercise continuous actions,
observation normalisation, or the GPU.

**Pendulum tuning result (2026-10-05).** One seed, 1,000 training episodes,
the same learning settings as CartPole, no observation normalisation or reward
scaling, stochastic evaluation over 10 episodes. Mean return by checkpoint:
-1,154.5 (0%), -755.9 (25%), -290.5 (50%), -378.9 (75%), -226.6 (100%). The
best final episodes reach -1.7, the worst -620.7, so the policy balances well
from some starting angles and does not yet swing up reliably from all of them.
Pendulum never terminates, so its success rate is 0% by definition. This shows
that continuous PPO learns end to end, including the latent-action storage and
the time-limit bootstrap that ends every Pendulum episode; it is one seed and
untuned, so the multi-seed standard run remains the formal validation.

**Pendulum standard result (2026-10-05).** Three seeds (14429 to 14431), 2,000
training episodes each, the same settings, stochastic evaluation over 50
episodes per checkpoint. Every seed learned: mean returns went from -1,190.9 to
-1,252.1 untrained to -197.6, -191.3, and -164.0, or -184.3 +/- 17.9 across
seeds, by 25% of training already between -221 and -268. Over the 150 final
evaluation episodes the median is -130.3 and 57% are better than -200; the
worst is -648.4, so some starting angles still defeat the swing-up. Seed 14431
fell back to -602.1 at 75% and recovered, the kind of instability untuned PPO
shows. This is the first standard validation of continuous PPO, and it
completes the validation of the batched PPO.
