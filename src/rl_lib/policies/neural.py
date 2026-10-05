"""Policies backed by neural-network distribution parameters."""

import math

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike

from rl_lib.networks import CategoricalPolicyNetwork, GaussianPolicyNetwork


class CategoricalPolicy:
    """Categorical action policy backed by a discrete policy network."""

    def __init__(self, network: CategoricalPolicyNetwork) -> None:
        self.network = network

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return their log-probabilities."""
        distribution = self._distribution(observation)
        action = distribution.sample()
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
        return torch.distributions.Categorical(logits=self.network(observation))


class SquashedGaussianPolicy:
    """Bound a diagonal Gaussian policy to a finite continuous action space."""

    def __init__(
        self,
        network: GaussianPolicyNetwork,
        action_low: ArrayLike,
        action_high: ArrayLike,
    ) -> None:

        action_low_array = np.asarray(action_low, dtype=np.float32)
        action_high_array = np.asarray(action_high, dtype=np.float32)
        expected_shape = (network.action_size,)
        if action_low_array.shape != expected_shape:
            raise ValueError("action_low must contain one value per action component")
        if action_high_array.shape != expected_shape:
            raise ValueError("action_high must contain one value per action component")
        if not np.all(np.isfinite(action_low_array)) or not np.all(
            np.isfinite(action_high_array)
        ):
            raise ValueError("Action bounds must be finite")
        if np.any(action_high_array <= action_low_array):
            raise ValueError(
                "Each upper action bound must be greater than its lower bound"
            )

        self.network = network
        scale = (action_high_array - action_low_array) / 2
        bias = (action_high_array + action_low_array) / 2

        self.scale = torch.as_tensor(scale, dtype=torch.float32)
        self.bias = torch.as_tensor(bias, dtype=torch.float32)

    def _to_environment_action(
        self,
        latent_action: torch.Tensor,
    ) -> torch.Tensor:
        # First map the unbounded Gaussian sample into the normalized range
        # [-1, 1], then map that range into each environment action bound.
        normalized_action = torch.tanh(latent_action)
        scale = self.scale.to(latent_action)
        bias = self.bias.to(latent_action)
        return scale * normalized_action + bias

    def _distribution(
        self,
        observation: torch.Tensor,
    ) -> torch.distributions.Normal:
        # The network defines one independent Normal distribution per action
        # component by predicting its mean and standard deviation.
        mean, std = self.network(observation)
        return torch.distributions.Normal(mean, std)

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
        scale = self.scale.to(latent_action)
        log_scale_jacobian = torch.log(scale)

        component_log_probabilities = (
            base_log_prob - log_tanh_jacobian - log_scale_jacobian
        )

        # The components form one joint action. Their densities multiply, so
        # their log-densities add to one value per observation.
        return component_log_probabilities.sum(dim=-1)

    def _entropy(
        self,
        distribution: torch.distributions.Normal,
    ) -> torch.Tensor:
        # The entropy after tanh has no simple closed-form expression. Estimate
        # it with one fresh action sampled from the current distribution.
        # rsample() keeps the sample differentiable so entropy regularization can
        # update both the means and standard deviations of the policy.
        latent_action = distribution.rsample()
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

        # Keep the raw sample for stable probability evaluation. sample(), unlike
        # rsample(), intentionally does not differentiate through random choice.
        latent_action = distribution.sample()

        # Only the bounded and rescaled action is sent to the environment.
        environment_action = self._to_environment_action(latent_action)

        # This remains differentiable through the distribution parameters, so a
        # policy-gradient loss can increase or decrease support for this action.
        log_probability = self._log_probability(distribution, latent_action)

        return environment_action, latent_action, log_probability
