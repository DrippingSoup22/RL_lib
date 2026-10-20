# Value normalisation

Added 2026-10-20, in `algorithms/policy_gradient/ppo.py`
(`PPO(value_normalization=, value_normalization_horizon=)`) and
`normalization.py` (`ObservationNormalizer(horizon=)`, `denormalize`, and
`clip_limit=math.inf`).

## Sources

- Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu, "The Surprising Effectiveness of
  PPO in Cooperative Multi-Agent Games", NeurIPS 2022 Datasets and Benchmarks
  (MAPPO): its Suggestion 1 is to standardise the critic's targets with
  running estimates of their mean and standard deviation, and to turn the
  critic's outputs back with the same estimates for advantage estimation.
- van Hasselt, Guez, Hessel, Mnih, Silver, "Learning values across many
  orders of magnitude", NeurIPS 2016 (PopArt): adaptive normalisation of value
  targets, which MAPPO's algorithm listing names for this step.
- Chan, Golub, LeVeque, "Updating Formulae and a Pairwise Algorithm for
  Computing Sample Variances", 1979: the batch update of the mean and
  variance, which the normaliser already used for observations.

The forgetting horizon is this library's addition, explained below.

## The idea

A critic is a neural network trained by regression on return targets, and
the size of those targets depends on the reward's units and the discount. A
reward of about 1 per step with discount 0.99 gives returns near 100; a
reward measured in hundreds gives returns of tens of thousands. Networks
learn best when their targets are about 1 in size: their starting weights,
the learning rate and gradient clipping all assume it. With large targets the
critic's first errors and gradients are huge, and it spends many updates just
reaching the right scale; with tiny targets it hardly learns at all.

Value normalisation lets the critic always learn targets of unit scale. Its
return targets are standardised: the running mean of all targets seen so far
is subtracted, and the result divided by their running standard deviation.
Whenever the critic's values are used, its outputs are turned back into
returns with the same mean and standard deviation.

## The update

For each critic, the normaliser keeps the running mean `μ` and standard
deviation `σ` of its return targets; with several reward channels, each
critic has its own (`docs/reward-channels.md`). The network's output `f(s)`
lives on the standardised scale, and its value in return units is

```text
V(s) = σ f(s) + μ
```

Each update then goes:

1. While collecting, `V(s)` in return units serves generalized advantage
   estimation, which computes the advantages `A` and return targets `G` in
   return units, as before.
2. The batch's return targets are merged into the running statistics first,
   so that they count towards their own standardisation.
3. The critic regresses on the standardised targets:
   `L = ½ mean (Ĝ − f(s))²`, with `Ĝ = (G − μ) / √(σ² + ε)`.
4. The actor is unchanged: its advantages come from step 1, in return units,
   and are normalised over the batch anyway.

When `μ` and `σ` change, the network's outputs, learned on the old scale, are
read on the new one, so its values move a little at that moment. PopArt
avoids this by also rescaling the network's last layer at every change; this
implementation does not, and the critic corrects the difference as it keeps
learning.

**Forgetting.** By default the statistics weigh every return target since the
start alike. But returns change as the policy learns: a policy that improves
earns more. After many updates the statistics hardly move any more, and they
describe a mixture dominated by old returns, so today's standardised targets
may lie far from unit scale. With a horizon `H`, before each update's batch is
merged the old statistics are made to count as at most `H − 1` batches. The
batch then moves the mean `1/H` of the way to its own mean `m`, and the
variance likewise, plus the spread between the two means:

```text
μ  ←  μ + (m − μ) / H
σ² ←  (1 − 1/H) σ² + (1/H) s² + (1/H) (1 − 1/H) (m − μ)²
```

where `s²` is the batch's own variance and `μ` on the right is the old mean.
This is an exponential moving average over about the last `H` updates.
`H = 0`, the default, never forgets.

## Implementation

- `ObservationNormalizer` now serves return targets as well as observations:
  `clip_limit=math.inf` turns clipping off, `denormalize` is the inverse of
  `normalize` without its clipping, and `horizon` sets the forgetting, in
  batches of statistics added. The horizon is saved by `state_dict`; a state
  saved before it existed restores with horizon 0.
- `PPO(value_normalization=False, value_normalization_horizon=0)` keeps one
  normaliser, `PPO.value_normalizer`, with one column per critic. Without
  value normalisation it is in mode `"none"` and changes nothing, and the
  horizon is not used.
- `sample_action(...).value` and `state_value` give values in return units,
  and `update` takes return targets in return units and standardises them
  itself, so a caller's rollout code is the same with or without value
  normalisation. The summary's `critic_loss` is measured on the standardised
  targets; `explained_variance` in return units.
- `PPO.state_dict()` saves the normaliser as `value_normalizer`; a state
  saved before value normalisation existed restores without it.

For example, a critic that learns standardised returns whose statistics
forget over about the last 100 updates:

```python
ppo = PPO(
    actor_network,
    actor_optimizer,
    critic_network,
    critic_optimizer,
    action_low=low,
    action_high=high,
    value_normalization=True,
    value_normalization_horizon=100,
)
```
