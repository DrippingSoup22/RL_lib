# Temporal action smoothness (CAPS)

Added 2026-10-09 for the Centipede project, in
`algorithms/policy_gradient/ppo.py` (`PPO(temporal_smoothness_coefficient=)`,
`update(next_observations=)`).

## Source

Mysore, Mabsout, Mancuso, Saenko, "Regularizing Action Policies for Smooth
Control with Reinforcement Learning", ICRA 2021 (arXiv:2012.06644): CAPS,
Conditioning for Action Policy Smoothness.

## The update

CAPS adds two regularization terms to the policy objective `J`:

```text
J_CAPS = J − λ_T L_T − λ_S L_S
L_T = D(π(s_t), π(s_{t+1}))        temporal: the next state's action stays close
L_S = D(π(s_t), π(s̄_t)),  s̄_t ~ N(s_t, σ)   spatial: nearby states, close actions
```

with the Euclidean distance `D(a1, a2) = ||a1 − a2||_2` between the actions.
They act on the policy itself rather than on the reward, so they need no
reward engineering and cannot be outweighed by large rewards. Applied to PPO,
TD3, SAC and DDPG on Gym tasks, CAPS made the actions smoother at about the
same reward (PPO on Ant-v2: smoothness 6.09 → 1.60 × 10⁻³, reward 3,735 →
4,257); on a real quadrotor, where PPO's motor commands oscillated at high
frequency, it cut the power used by almost 80%. The authors also explain why
a filter on the actions is no substitute: it breaks the Markov property the
policy was trained under.

Only the temporal term is implemented here. For PPO with a squashed Gaussian
policy, `π(s)` is the mean action, `tanh(μ(s))` scaled to the bounds, and the
actor's loss in each minibatch becomes

```text
actor loss = −mean(clipped surrogate) − c_entropy × mean(entropy)
             + λ_T × mean_i ||π(s_i) − π(s'_i)||_2
```

where `s'_i` is the observation after `s_i`. A tiny constant under the square
root keeps the distance differentiable where the two actions are equal. The
term needs the next observations, so `update` takes `next_observations`
whenever `temporal_smoothness_coefficient` is above 0; for a step that ended
an episode, the next observation is the episode's last one, which still
follows the step. Categorical PPO has no mean action and refuses the term.
