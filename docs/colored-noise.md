# Colored exploration noise

Added 2026-10-09 for the Centipede project's smoothness tests, in
`policies/neural.py` (`colored_noise`, `SquashedGaussianPolicy(noise_beta=,
noise_sequence_steps=)`) and passed through `PPO(noise_beta=,
noise_sequence_steps=)`.

## Sources

- Eberhard, Hollenstein, Pinneri, Martius, "Pink Noise Is All You Need:
  Colored Noise Exploration in Deep Reinforcement Learning", ICLR 2023.
- Hollenstein, Martius, Piater, "Colored Noise in PPO: Improved Exploration
  and Performance through Correlated Action Sampling", AAAI 2024.
- Timmer, König, "On generating power law noise", Astronomy and Astrophysics
  300, 1995: the generation method, as implemented in Patzelt's
  `colorednoise` package (MIT license), which Eberhard et al. use.

## The update

A squashed Gaussian policy samples a latent action `u = μ(s) + σ(s) ε` and
sends `tanh(u)`, scaled to the bounds. Normally `ε ~ N(0, I)` is drawn afresh
at every step: white noise, whose power is the same at every frequency, so the
actions jitter from step to step.

Colored noise keeps each step's `ε_t ~ N(0, 1)` per component but correlates
it over time: the sequence `ε_1 … ε_T` of one component has power spectral
density `S(f) ∝ 1/f^β`. `β = 0` is white noise, `β = 1` pink noise and `β = 2`
red (Brownian-like) noise; the larger `β`, the more of the power lies at low
frequencies and the more slowly the noise drifts. For 1,000-step sequences:

| β | Correlation with the step before | 10 steps before | Change per step, RMS |
| --- | --- | --- | --- |
| 0 | 0.00 | 0.00 | 1.41 |
| 0.5 | 0.36 | 0.09 | 1.14 |
| 1 | 0.77 | 0.45 | 0.70 |
| 2 | 1.00 | 0.96 | 0.10 |

Nothing else changes. The log-probability of a sampled latent action is the
same Gaussian density as before, because each step's noise still has a
standard normal distribution; PPO's ratio and clipping are untouched. Only
the joint distribution of consecutive actions differs: the collected data are
"asymptotically on-policy" at each step (Hollenstein et al.), and within one
sequence the actions share a slowly varying offset, larger for larger `β`.

**Generation (Timmer and König).** For a sequence of `T` steps, take the
real FFT frequencies `f_k = k/T`, `k = 0 … T/2`, with `f_0` replaced by `f_1`.
Draw each Fourier coefficient's real and imaginary parts from `N(0, s_k²)`,
`s_k = f_k^(−β/2)`; the constant term (and for even `T` the highest
frequency's) is real, with `√2` times the spread. The inverse real FFT gives
the sequence, divided by its theoretical standard deviation
`2 √(Σ_{k≥1} w_k²) / T`, where `w_k = s_k`, halved for the highest frequency
when `T` is even. Each sequence then has unit variance about its own mean;
counting the random mean too, a step's variance is 1.01 for `β = 0.5`, 1.07
for `β = 1` and 1.34 for `β = 2` at `T = 1000`.

## Implementation

- `colored_noise(beta, shape, steps, generator, device, dtype)` returns
  `(steps, *shape)` sequences, one per element of `shape`, drawn from the
  given generator.
- `SquashedGaussianPolicy(noise_beta=0.0, noise_sequence_steps=1000)`: with
  `noise_beta` 0, `sample` draws white noise exactly as before. Above 0,
  `sample` takes the next step of `(noise_sequence_steps, batch, actions)`
  sequences, drawing new ones when they are used up or the batch size
  changes; the sequences do not restart with episodes. The default, 1,000
  steps, is Hollenstein et al.'s, whose tasks' episodes last up to 1,000
  steps; Eberhard et al.'s code sets the length to the task's episode length.
  Shorter sequences hold less of the slowest drift: at β = 1, 256-step
  sequences correlate 0.72 with the step before and 0.32 ten steps before,
  against 0.77 and 0.45 at 1,000 steps. The entropy estimate keeps drawing
  white noise: it estimates the policy's entropy, which colored noise does not
  change.
- The current sequences are not part of `PPO.state_dict()`: a run continued
  from a checkpoint starts new sequences.
