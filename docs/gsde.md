# Generalized state-dependent exploration (gSDE)

Added 2026-10-20, in `networks/policy.py`
(`GaussianPolicyNetwork(std_mode="gsde")`) and `policies/neural.py`
(`SquashedGaussianPolicy(exploration_matrix_steps=)`), and passed through
`PPO(exploration_matrix_steps=)`.

## Sources

- Raffin, Kober, Stulp, "Smooth Exploration for Robotic Reinforcement
  Learning", CoRL 2021 (arXiv:2005.05719): gSDE.
- Rückstieß, Felder, Schmidhuber, "State-Dependent Exploration for Policy
  Gradient Methods", ECML PKDD 2008: state-dependent exploration, which gSDE
  generalizes.
- Stable-Baselines3's `StateDependentNoiseDistribution`, Raffin et al.'s own
  implementation: the small floor added to the variance.

## The idea

A Gaussian policy explores by adding noise to its mean action:
`u = μ(s) + σ ε`. Usually `ε` is drawn afresh at every step, so the noise of
one step has nothing to do with the noise of the next. On a robot, or a
simulated body driven by motors, this makes the actions shake from step to
step: the motors receive a new random push many times per second, most of
which the next push cancels.

gSDE makes the noise a function of the state instead. For a while, the policy
holds one random matrix, and the noise is that matrix applied to what the
network sees in the state. The same state then always gets the same noise,
and nearby states get similar noise. As the body moves smoothly through its
states, the noise changes smoothly with them. Every few steps the matrix is
drawn again, so that exploration still tries something new.

## The update

Let `z(s)` be the features the mean's output layer reads: the last hidden
layer, or the observation itself if there are no hidden layers, `F` values.
Here they are scaled to unit length, `v = z / |z|`. Raffin et al. use `z`
itself; scaling it means the size of the noise no longer depends on how large
the features happen to be, so the starting spread keeps its meaning.

The exploration matrix `E` has one entry per feature `i` and action component
`j`, each drawn from a normal distribution with its own learned spread:
`E[i, j] ~ N(0, S[i, j]²)`, with `S = exp(log_std)`, an `F × A` matrix of
parameters. The noise and the latent action are

```text
n_j(s) = Σ_i v_i(s) E[i, j]
u_j    = μ_j(s) + n_j(s)
```

For one fixed state, `n_j` is a sum of independent normal numbers, so it is
normal itself, with mean 0 and variance

```text
σ_j(s)² = Σ_i v_i(s)² S[i, j]²
```

Different action components use different columns of `E`, so they stay
independent. The policy's distribution in a state is therefore the usual
diagonal Gaussian `N(μ(s), diag σ(s)²)`, and the log-probability that PPO's
ratio needs is computed exactly as before, as Raffin et al. do. A floor of
`10⁻⁶` is added to the variance, as in Stable-Baselines3, so that a state
whose features are all zero still has a proper distribution.

When every spread equals `s`, the variance is `s² Σ_i v_i² = s²`, because `v`
has unit length: the standard deviation is `s` in every state. So
`initial_std` starts the policy with the same spread as the other modes do.

**Learning.** The spreads `S` learn through `σ(s)`: the log-probability's
gradient tells each `S[i, j]` whether exploring more or less along that
feature paid off. The features are treated as data in the noise and in
`σ(s)`, so the hidden layers learn only through the mean, as in Raffin et
al.'s default.

**How long a matrix lasts.** Each row of a batch (each environment) has its
own matrix, held for `n` calls of `sample`, then drawn again. With `n = 1` a
new matrix comes at every step: the noise's size still depends on the state,
but it no longer follows it from step to step. A larger `n` gives smoother,
more consistent exploration, at the cost of trying fewer different
directions. Raffin et al. treat `n` as a setting to tune.

## Implementation

- `GaussianPolicyNetwork(std_mode="gsde")`: `log_std` is the `F × A` matrix
  of log spreads, starting at `log(initial_std)`. `forward` returns the mean
  and `σ(s)`; `forward_with_features` also returns the unit-length features
  `v`; `state_dependent_noise(v, Z)` computes `n = v (S ∘ Z)` from standard
  normal matrices `Z`. The hidden layers can be ReLU or tanh; with ReLU, a
  state where every feature is zero gets almost no noise.
- `SquashedGaussianPolicy(exploration_matrix_steps=1)`: `sample` draws `Z`,
  one `F × A` matrix per row, from the policy's generator, keeps it for that
  many calls, and draws new ones after them or when the number of rows
  changes. The noise is divided by `σ(s)` to make it standard normal, so the
  rest of sampling is shared with white and colored noise. gSDE cannot be
  combined with colored noise (`noise_beta`). The entropy estimate keeps
  drawing white noise from each state's Gaussian.
- `PPO(exploration_matrix_steps=1)` passes the setting to the policy. The
  matrices are not part of `PPO.state_dict()`: a run continued from a
  checkpoint draws new ones.
- Memory: the matrices hold `rows × F × A` numbers, for example 2 million
  (8 MB) for 4,096 environments, 64 features and 8 actions.
