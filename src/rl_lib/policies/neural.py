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
    """

    def __init__(
        self,
        network: GaussianPolicyNetwork,
        action_low: torch.Tensor | Sequence[float],
        action_high: torch.Tensor | Sequence[float],
        *,
        generator: torch.Generator | None = None,
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

        self.network = network
        self.generator = generator
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
    ) -> torch.Tensor:
        # The mean plus the spread times standard normal noise from this
        # policy's generator. Written this way, the draw stays differentiable
        # through the mean and spread, like Normal.rsample.
        noise = torch.randn(
            distribution.loc.shape,
            dtype=distribution.loc.dtype,
            device=distribution.loc.device,
            generator=self.generator,
        )
        return distribution.loc + distribution.scale * noise

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
        # network.
        latent_action = self._draw_latent_action(distribution).detach()

        # Only the bounded and rescaled action is sent to the environment.
        environment_action = self._to_environment_action(latent_action)

        # This remains differentiable through the distribution parameters, so a
        # policy-gradient loss can increase or decrease support for this action.
        log_probability = self._log_probability(distribution, latent_action)

        return environment_action, latent_action, log_probability
