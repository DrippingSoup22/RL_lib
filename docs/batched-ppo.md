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
| PPO `seed`, `rng` | `seed`, `generator` | First renamed `shuffle_seed` and `shuffle_generator`; since the second round the generator drives everything random PPO does. |
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

## Second round: closing the PPO path

A review of everything PPO uses, made before the Centipede builds its agents
on it, found that the batched PPO does no NumPy work while it acts or learns,
but still had four shortcomings. They were measured on the local MX330 GPU
(PyTorch 2.14.1), with PyTorch's sync-debug mode, which reports every point
where the CPU stops to wait for the GPU.

- **Hidden waits for the GPU.** `sample_action` waited six times, each
  update minibatch six times, and a deterministic `select_action` twice. Two
  causes: PyTorch's distributions check their arguments by default, and every
  check reads a GPU value on the CPU; and `SquashedGaussianPolicy` kept its
  action scale and bias on the CPU and copied them to the GPU on every call.
  Removing them made a minibatch 4% faster and a `sample_action` of 1,024
  observations 18% faster (1.29 to 1.06 ms).
- **No random generator of its own.** Only the minibatch order had its own
  generator; actions and the continuous entropy estimate were drawn from
  PyTorch's global generator, which any other code in the process also draws
  from. Several PPO agents in one program could not each own a random sequence,
  and a checkpoint could not restore one.
- **No checkpoints.** A user had to collect both networks, both optimizers, and
  the generator separately.
- **NumPy in the constructors.** Settings and action bounds were checked with
  NumPy, so bounds given as GPU tensors failed.

The measurements also showed what does not need changing. A minibatch costs
about 3 ms on the CPU and 8 ms on the MX330 whatever its size up to 4,096
samples, so the time of an update depends mostly on how many minibatches it
takes; that is the user's choice of minibatch size, not a library fault. One
`sample_action` takes about 1 ms, so evaluating several agents' networks in
one operation is not worth its complexity next to a slow simulation.

| Part | Change |
| --- | --- |
| `policies/neural.py` | Distributions are built without argument checks. The action scale and bias are created once on the network's device, from bounds given as a tensor or a sequence and checked once on the CPU. Both policies take an optional `generator` and draw every random number from it, falling back to PyTorch's global generator without one. |
| `algorithms/policy_gradient/ppo.py` | `seed` creates one `generator`, used for sampling actions, estimating the continuous entropy, and shuffling minibatches. `state_dict()` and `load_state_dict()` save and restore both networks, both optimizers, and the generator; the generator is restored only on the same kind of device, since CPU and CUDA generators draw different sequences. `PPOUpdateSummary` adds `explained_variance`, how much of the returns' variance the values that collected the batch explained. |
| `normalization.py` | `load_state_dict()` restores saved statistics into an existing normaliser whose size and mode match; `from_state_dict()` builds a new one with it. |
| `networks/policy.py`, `optimization/gradient_clipping.py` | Settings are checked with `math` instead of NumPy. |
| `experiments/policy_gradient/` | One helper seeds an episode's actions: PPO's generator, and PyTorch's global one for the other algorithms. |

Without argument checks, a non-finite network output no longer raises an error
inside a distribution. Users check the update summary once per update instead,
as the experiment runner does. Because actions are now drawn from PPO's own
generator, a run with the same seeds no longer repeats the earlier runs'
episodes, so the two standard runs are repeated.

## Not included

The other algorithms and families, TRPO, a vectorised Gymnasium runner, rollout
storage (each user of PPO keeps its own), and evaluating several agents'
networks in one operation (measured above as not worth it).

## Validation

- Unit tests: GAE matches hand-computed values, including a block in which
  environments end by termination, by time limit, and by the window ending, at
  different steps; the normaliser gives the same statistics whether a batch is
  added whole or in parts; a batch gives the same log-probabilities and values
  as its rows one at a time; an update runs on CUDA when a GPU is available.
- Second round, unit tests: on a GPU, acting and a minibatch step never make
  the CPU wait; the same seed repeats the same actions whatever the global
  generator does; a PPO restored from `state_dict()` then acts and learns
  exactly like the original; the explained variance matches its definition;
  a saved normaliser restores into an existing one.
- End to end: continuous PPO on `Pendulum-v1`, the first standard validation of
  continuous PPO, and a repeat of the discrete `CartPole-v1` standard run to
  confirm that nothing regressed. Both are repeated once after the second round,
  since its random sequences differ.

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
| Second round: code, tests, documentation | Done |
| Second round: CartPole standard run repeated | Passed (`runs/CartPole-v1/policy_gradient/ppo/20261005T190913000252Z`) |
| Second round: Pendulum standard run repeated | Passed (`runs/Pendulum-v1/policy_gradient/ppo/20261005T192332852663Z`) |

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

**Second-round results (2026-10-05).** Both standard runs were repeated with
the same 34 recorded settings and paired seeds, on PyTorch 2.7.1. The untrained
returns at the 0% checkpoint are identical to the earlier runs (CartPole 20.3,
18.3, 22.1; Pendulum -1,214.8, -1,252.1, -1,190.9): seeding PPO's own generator
with an episode's action seed draws the same actions as seeding the global
generator did. Training then differs, as expected, because minibatches are now
shuffled, and the continuous entropy estimated, from that same generator.
CartPole: all three seeds reached a frozen-policy return of 500 with 100%
success, every seed by 50% of training (seed 932106 at 25%: 354, against 295
before). Pendulum: final mean returns of -237.5, -180.2, and -162.6, or
-193.5 +/- 39.2 across seeds (before: -184.3 +/- 17.9); over the 150 final
evaluation episodes the median is -127.5 (before -130.3) and 60% are better
than -200 (before 57%). Seed 14431 collapsed to -1,300.4 at 50% and recovered,
the same kind of instability as before (-602.1 at 75%). Both runs pass; the
second round changed no learning behaviour beyond the random sequences.
