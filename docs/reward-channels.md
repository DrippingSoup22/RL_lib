# Several reward channels

Added 2026-10-20, in `algorithms/policy_gradient/ppo.py` (`PPO` given a
sequence of critics and `advantage_weights=`) and
`algorithms/policy_gradient/advantages.py` (`generalized_advantage_estimates`
with a channel dimension).

## Source

- van Seijen, Fatemi, Romoff, Laroche, Barnes, Tsang, "Hybrid Reward
  Architecture for Reinforcement Learning", NeurIPS 2017: split the reward
  into parts and learn a separate value function for each. They apply it to
  Q-learning; here the same idea is applied to PPO's critic.

How the channels' advantages are combined for the actor, each normalised on
its own and then weighted, is this library's choice, explained below; it is
not taken from the paper.

## The idea

Many tasks pay a reward made of several parts, for example progress towards
a goal and a cost for the energy used. With one critic, the critic must
predict the value of the sum. The parts may have very different scales and
depend on different things, and a small but important part is easily drowned
in the variation of a large one: the critic learns it badly, and the actor's
advantages carry little of it.

With several reward channels, each part is a channel with a critic of its
own. Each critic learns a simpler function, the value of one part, from
what that part depends on. The actor then learns from one advantage built
from all the channels.

## The update

The task gives `K` channels of rewards `r_t^k`, and each channel `k` has a
critic `V_k`. Generalized advantage estimation runs on each channel
separately, with the same discount `γ`, the same `λ`, and the same episode
ends:

```text
δ_t^k = r_t^k + γ (1 − terminated_t) V_k(s_{t+1}) − V_k(s_t)
A_t^k = δ_t^k + γ λ (1 − ended_t) A_{t+1}^k
G_t^k = A_t^k + V_k(s_t)                   return target of critic k
```

**Actor.** Each channel's advantages are normalised over the whole batch on
their own, to mean 0 and standard deviation 1, and then added with fixed
weights `w_k`:

```text
Â_t = Σ_k w_k (A_t^k − mean(A^k)) / std(A^k)
```

PPO's clipped objective then uses `Â_t` exactly as it uses the single
advantage. Because every channel has the same spread before weighting, the
weights alone set how much each channel counts in the actor's gradient,
whatever the units of its reward. With one channel, this is PPO's usual
advantage normalisation.

This has a cost worth knowing: the actor no longer follows the plain sum of
the rewards. A channel's rewards count with the weight `w_k / std(A^k)`, and
the spread changes as learning changes the returns. A channel whose
advantages hardly vary is magnified to the same spread as the others, its
noise included.

**Critics.** Each critic learns its own return targets, and their losses add
up:

```text
L_critic = Σ_k ½ mean_t (G_t^k − V_k(s_t))²
```

The critics share no parameters, so each receives only the gradient of its
own error. With gradient clipping, every network's gradients are clipped on
their own.

## Implementation

- `PPO(actor_network, actor_optimizer, [V_1, …, V_K], critic_optimizer,
  advantage_weights=[w_1, …, w_K])`: the critics are given as a sequence, and
  one optimizer updates all their parameters. The weights default to 1 each.
  A single critic, not in a sequence, is ordinary PPO and takes no weights;
  a sequence of one critic gives every tensor a channel column of size 1.
- With several critics, `sample_action(...).value` and `state_value` return
  `(B, K)`: each critic's value, one column per channel.
- `generalized_advantage_estimates` takes `rewards`, `values` and
  `next_values` of shape `(T, W, K)` (or `(T, K)`), while `terminated` and
  `episode_ended` keep the steps' shape `(T, W)` and apply to every channel.
  It returns advantages and return targets of shape `(T, W, K)`.
- `update` takes `advantages` and `return_targets` of shape `(N, K)`.
- `PPOUpdateSummary`: `critic_loss` is the sum of the critics' losses, and
  `explained_variance` a tuple with each critic's, in order.
- `state_dict()` saves the critics as a list of state dicts, in order.

For example, two channels whose advantages count 1 and 0.5:

```python
critics = [StateValueNetwork(observation_size) for _ in range(2)]
ppo = PPO(
    actor_network,
    torch.optim.Adam(actor_network.parameters(), lr=3e-4),
    critics,
    torch.optim.Adam(
        [parameter for critic in critics for parameter in critic.parameters()],
        lr=3e-4,
    ),
    action_low=low,
    action_high=high,
    advantage_weights=[1.0, 0.5],
)
```
