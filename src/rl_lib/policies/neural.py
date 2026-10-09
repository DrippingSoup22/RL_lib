"""Policies backed by neural-network distribution parameters.

Both policies draw their random numbers from the ``generator`` they are given,
so that an algorithm can own its random sequence; without one they use
PyTorch's global generator. Their distributions skip PyTorch's argument checks:
each check reads a GPU value on the CPU, which makes the CPU wait for the GPU on
every call. Callers check results once instead, for example an update's losses.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from rl_lib.networks import CategoricalPolicyNetwork, GaussianPolicyNetwork

# How many steps of colored noise are drawn at a time unless a policy is told
# otherwise, as in Hollenstein, Martius and Piater (AAAI 2024).
COLORED_NOISE_STEPS = 1000


def colored_noise(
    beta: float,
    shape: torch.Size,
    steps: int,
    *,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Standard normal noise whose power falls as 1/f^beta along time.

    Returns ``(steps, *shape)``: an independent sequence through time for
    every element of ``shape``. ``beta`` 0 gives white noise, 1 pink noise and
    2 red noise; the larger it is, the more slowly the noise drifts. The method
    is Timmer and König (1995), as in Patzelt's ``colorednoise`` package (MIT
    license), which Eberhard et al. (ICLR 2023) use: each Fourier coefficient
    is a complex normal number scaled by f^(-beta/2), the lowest frequency's
    scale standing in for f = 0, and the inverse transform gives the sequence,
    divided by its theoretical standard deviation.
    """
    frequencies = torch.fft.rfftfreq(steps, device=device, dtype=dtype)
    frequencies[0] = frequencies[1]
    scale = frequencies ** (-beta / 2)
    # The standard deviation the coefficients give the sequence. At an even
    # length, the highest frequency's coefficient counts once, not twice.
    weights = scale[1:].clone()
    weights[-1] *= (1 + steps % 2) / 2
    standard_deviation = 2 * torch.sqrt(torch.sum(weights**2)) / steps

    size = (*shape, len(frequencies))
    real = scale * torch.randn(size, generator=generator, device=device, dtype=dtype)
    imaginary = scale * torch.randn(
        size, generator=generator, device=device, dtype=dtype
    )
    # The constant term, and at an even length the highest frequency's, are
    # real numbers, given the whole power of a complex one.
    imaginary[..., 0] = 0
    real[..., 0] *= math.sqrt(2)
    if steps % 2 == 0:
        imaginary[..., -1] = 0
        real[..., -1] *= math.sqrt(2)
    sequences = torch.fft.irfft(torch.complex(real, imaginary), n=steps)
    return (sequences / standard_deviation).movedim(-1, 0).contiguous()


class CategoricalPolicy:
    """Categorical action policy backed by a discrete policy network."""

    def __init__(
        self,
        network: CategoricalPolicyNetwork,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.network = network
        self.generator = generator

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return their log-probabilities."""
        distribution = self._distribution(observation)
        # The same draw as Categorical.sample, but from this policy's generator.
        probabilities = distribution.probs.reshape(-1, self.network.number_of_actions)
        action = torch.multinomial(
            probabilities, 1, replacement=True, generator=self.generator
        ).reshape(distribution.batch_shape)
        return action, distribution.log_prob(action)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return log-probability and entropy for supplied actions."""
        distribution = self._distribution(observations)
        return distribution.log_prob(actions), distribution.entropy()

    def entropy(self, observations: torch.Tensor) -> torch.Tensor:
        """Return categorical entropy for each observation."""
        return self._distribution(observations).entropy()

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the action with the largest categorical preference."""
        return torch.argmax(self.network(observation), dim=-1)

    def _distribution(
        self,
        observation: torch.Tensor,
    ) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(
            logits=self.network(observation), validate_args=False
        )


class SquashedGaussianPolicy:
    """Bound a diagonal Gaussian policy to a finite continuous action space.

    Create it after moving the network to its device: the action scale and bias
    are placed on that device once, here.

    ``noise_beta`` colors the noise of sampled actions: 0, the default, draws
    fresh white noise for every sample; above 0, each row and action component
    of the samples follows its own colored-noise sequence (``colored_noise``)
    from one call of ``sample`` to the next, so that exploration moves smoothly
    instead of jittering. Each sample's noise is still standard normal, so its
    log-probability is unchanged (Hollenstein, Martius and Piater, AAAI 2024).
    The sequences are drawn ``noise_sequence_steps`` steps at a time, and drawn
    afresh when the number of rows changes. The default, 1,000 steps, is
    Hollenstein et al.'s; Eberhard et al.'s code (ICLR 2023) makes a sequence
    as long as the task's episodes.
    """

    def __init__(
        self,
        network: GaussianPolicyNetwork,
        action_low: torch.Tensor | Sequence[float],
        action_high: torch.Tensor | Sequence[float],
        *,
        generator: torch.Generator | None = None,
        noise_beta: float = 0.0,
        noise_sequence_steps: int = COLORED_NOISE_STEPS,
    ) -> None:
        # The bounds come from outside, usually an environment's action space,
        # so they are checked once, here, on the CPU.
        low = torch.as_tensor(action_low, dtype=torch.float32, device="cpu")
        high = torch.as_tensor(action_high, dtype=torch.float32, device="cpu")
        expected_shape = (network.action_size,)
        if low.shape != expected_shape:
            raise ValueError("action_low must contain one value per action component")
        if high.shape != expected_shape:
            raise ValueError("action_high must contain one value per action component")
        if not (torch.isfinite(low).all() and torch.isfinite(high).all()):
            raise ValueError("Action bounds must be finite")
        if torch.any(high <= low):
            raise ValueError(
                "Each upper action bound must be greater than its lower bound"
            )
        if not math.isfinite(noise_beta) or noise_beta < 0:
            raise ValueError("noise_beta must be finite and nonnegative")
        if noise_sequence_steps < 2:
            raise ValueError("noise_sequence_steps must be at least 2")

        self.network = network
        self.generator = generator
        self.noise_beta = noise_beta
        self.noise_sequence_steps = noise_sequence_steps
        # The colored-noise sequences being used, and how many steps of them.
        self._noise_sequences: torch.Tensor | None = None
        self._noise_steps_used = 0
        device = next(network.parameters()).device
        self.scale = ((high - low) / 2).to(device)
        self.bias = ((high + low) / 2).to(device)
        # The density correction for scaling [-1, 1] to the bounds never changes.
        self.log_scale = torch.log(self.scale)

    def _to_environment_action(
        self,
        latent_action: torch.Tensor,
    ) -> torch.Tensor:
        # First map the unbounded Gaussian sample into the normalized range
        # [-1, 1], then map that range into each environment action bound.
        return self.scale * torch.tanh(latent_action) + self.bias

    def _distribution(
        self,
        observation: torch.Tensor,
    ) -> torch.distributions.Normal:
        # The network defines one independent Normal distribution per action
        # component by predicting its mean and standard deviation.
        mean, std = self.network(observation)
        return torch.distributions.Normal(mean, std, validate_args=False)

    def _draw_latent_action(
        self,
        distribution: torch.distributions.Normal,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The mean plus the spread times standard normal noise, fresh from this
        # policy's generator unless given. Written this way, the draw stays
        # differentiable through the mean and spread, like Normal.rsample.
        if noise is None:
            noise = torch.randn(
                distribution.loc.shape,
                dtype=distribution.loc.dtype,
                device=distribution.loc.device,
                generator=self.generator,
            )
        return distribution.loc + distribution.scale * noise

    def _next_colored_noise(self, mean: torch.Tensor) -> torch.Tensor:
        # One step of the colored-noise sequences, one sequence per element of
        # the mean, drawing new ones when they are used up or the shape changes.
        sequences = self._noise_sequences
        if (
            sequences is None
            or sequences.shape[1:] != mean.shape
            or self._noise_steps_used == len(sequences)
        ):
            sequences = self._noise_sequences = colored_noise(
                self.noise_beta,
                mean.shape,
                self.noise_sequence_steps,
                generator=self.generator,
                device=mean.device,
                dtype=mean.dtype,
            )
            self._noise_steps_used = 0
        noise = sequences[self._noise_steps_used]
        self._noise_steps_used += 1
        return noise

    def _log_probability(
        self,
        distribution: torch.distributions.Normal,
        latent_action: torch.Tensor,
    ) -> torch.Tensor:
        # Start with the density of each raw sample under its Normal component.
        base_log_prob = distribution.log_prob(latent_action)

        # Correct that density for tanh compressing the unbounded latent space.
        # This stable identity avoids computing log(1 - tanh(z) ** 2) directly.
        log_tanh_jacobian = 2.0 * (
            math.log(2.0) - latent_action - nn.functional.softplus(-2.0 * latent_action)
        )

        # Correct once more for scaling [-1, 1] to the environment's bounds.
        component_log_probabilities = base_log_prob - log_tanh_jacobian - self.log_scale

        # The components form one joint action. Their densities multiply, so
        # their log-densities add to one value per observation.
        return component_log_probabilities.sum(dim=-1)

    def _entropy(
        self,
        distribution: torch.distributions.Normal,
    ) -> torch.Tensor:
        # The entropy after tanh has no simple closed-form expression. Estimate
        # it with one fresh action sampled from the current distribution. The
        # draw keeps gradients, so entropy regularization can update both the
        # means and standard deviations of the policy.
        latent_action = self._draw_latent_action(distribution)
        return -self._log_probability(distribution, latent_action)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate stored latent actions under the current policy."""
        distribution = self._distribution(observations)

        # Reuse the exact latent actions collected earlier. This is especially
        # important for PPO, which compares their old and new log-probabilities.
        log_probabilities = self._log_probability(distribution, latent_actions)

        # Entropy measures current exploration and uses a separate fresh sample;
        # the stored actions may have come from an older version of the policy.
        entropy = self._entropy(distribution)
        return log_probabilities, entropy

    def entropy(self, observations: torch.Tensor) -> torch.Tensor:
        """Estimate the squashed policy entropy for each observation."""
        return self._entropy(self._distribution(observations))

    def deterministic_action(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        # Deterministic deployment uses the distribution's center instead of a
        # random sample, then applies the same bounds as stochastic actions.
        mean, _ = self.network(observation)
        return self._to_environment_action(mean)

    def sample(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observation)

        # Keep the raw sample for stable probability evaluation. It is detached,
        # like Normal.sample: the sampled action is data, not a function of the
        # network. Only these samples use colored noise; the entropy estimate
        # always draws white noise.
        noise = self._next_colored_noise(distribution.loc) if self.noise_beta else None
        latent_action = self._draw_latent_action(distribution, noise).detach()

        # Only the bounded and rescaled action is sent to the environment.
        environment_action = self._to_environment_action(latent_action)

        # This remains differentiable through the distribution parameters, so a
        # policy-gradient loss can increase or decrease support for this action.
        log_probability = self._log_probability(distribution, latent_action)

        return environment_action, latent_action, log_probability
