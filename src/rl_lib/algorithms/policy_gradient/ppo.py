"""Clipped proximal policy optimization, on batches of tensors."""

import math
from dataclasses import dataclass

import torch
from numpy.typing import ArrayLike

from rl_lib.networks import (
    CategoricalPolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)
from rl_lib.optimization import clip_gradients, validate_max_gradient_norm
from rl_lib.policies import CategoricalPolicy, SquashedGaussianPolicy


@dataclass(frozen=True)
class PPOActionSample:
    """One sampled action per observation, with what learning keeps about it.

    ``environment_action`` is what the environment receives. ``policy_action``
    is what ``update`` re-evaluates later: the latent Gaussian sample for
    continuous actions, and the action itself for categorical ones.
    ``log_probability`` is the sampling policy's log-probability of it and
    ``value`` the critic's value of the observation, both ``(B,)``.
    """

    environment_action: torch.Tensor
    policy_action: torch.Tensor
    log_probability: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class PPOUpdateSummary:
    """The means, over all minibatches of one update, of their measurements."""

    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


class PPO:
    """PPO with a separate actor and critic, for categorical or bounded actions.

    Everything works on batches of tensors on the networks' device: the caller
    moves the networks there before creating their optimizers, and passes
    observations and stored data already on it. Nothing is converted to Python
    numbers inside a loop, so a GPU is never kept waiting.
    """

    def __init__(
        self,
        actor_network: CategoricalPolicyNetwork | GaussianPolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        critic_network: StateValueNetwork,
        critic_optimizer: torch.optim.Optimizer,
        clip_ratio: float = 0.2,
        entropy_coefficient: float = 0.0,
        shuffle_seed: int | None = None,
        *,
        max_gradient_norm: float | None = None,
        action_low: ArrayLike | None = None,
        action_high: ArrayLike | None = None,
    ) -> None:
        """Check the settings once and choose the policy from the actor network.

        ``shuffle_seed`` fixes the order in which minibatches are drawn; without
        it the order is random. Continuous actors need ``action_low`` and
        ``action_high``; categorical actors must not get them.
        """
        if actor_network.observation_size != critic_network.observation_size:
            raise ValueError("Actor and critic networks must have the same input size")
        if not math.isfinite(clip_ratio) or not 0 < clip_ratio < 1:
            raise ValueError("Clip ratio must be finite and in (0, 1)")
        if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0:
            raise ValueError("Entropy coefficient must be finite and nonnegative")

        self.actor_network = actor_network
        self.actor_optimizer = actor_optimizer
        self.critic_network = critic_network
        self.critic_optimizer = critic_optimizer
        self.clip_ratio = clip_ratio
        self.entropy_coefficient = entropy_coefficient
        self.max_gradient_norm = validate_max_gradient_norm(max_gradient_norm)

        # The networks' device is where all data must be, and where the
        # minibatch order is drawn.
        self.device = next(actor_network.parameters()).device
        self.shuffle_generator = torch.Generator(device=self.device)
        if shuffle_seed is None:
            self.shuffle_generator.seed()
        else:
            self.shuffle_generator.manual_seed(shuffle_seed)

        self.policy: CategoricalPolicy | SquashedGaussianPolicy
        if isinstance(actor_network, CategoricalPolicyNetwork):
            if action_low is not None or action_high is not None:
                raise ValueError("Categorical PPO must not receive action bounds")
            self.policy = CategoricalPolicy(actor_network)
        elif isinstance(actor_network, GaussianPolicyNetwork):
            if action_low is None or action_high is None:
                raise ValueError(
                    "Continuous PPO requires lower and upper action bounds"
                )
            self.policy = SquashedGaussianPolicy(actor_network, action_low, action_high)
        else:
            raise TypeError("Actor network must be categorical or Gaussian")

    def sample_action(self, observations: torch.Tensor) -> PPOActionSample:
        """Sample one action per observation, for collecting data to learn from.

        ``observations`` is ``(B, observation_size)``. Nothing keeps gradients:
        the sample is a fixed record that ``update`` later compares with the
        changed policy.
        """
        with torch.no_grad():
            value = self.critic_network(observations)
            if isinstance(self.policy, CategoricalPolicy):
                action, log_probability = self.policy.sample(observations)
                # A categorical action is re-evaluated exactly as it was sent.
                environment_action = policy_action = action
            else:
                environment_action, policy_action, log_probability = self.policy.sample(
                    observations
                )
        return PPOActionSample(
            environment_action=environment_action,
            policy_action=policy_action,
            log_probability=log_probability,
            value=value,
        )

    def select_action(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """One environment action per observation, without the critic.

        For evaluation: ``deterministic`` takes the most likely categorical
        action, or the squashed Gaussian mean, instead of sampling.
        """
        with torch.no_grad():
            if deterministic:
                return self.policy.deterministic_action(observations)
            if isinstance(self.policy, CategoricalPolicy):
                action, _ = self.policy.sample(observations)
                return action
            environment_action, _, _ = self.policy.sample(observations)
            return environment_action

    def state_value(self, observations: torch.Tensor) -> torch.Tensor:
        """The critic's value of each observation, ``(B,)``, for bootstrapping."""
        with torch.no_grad():
            return self.critic_network(observations)

    def update(
        self,
        observations: torch.Tensor,
        policy_actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        return_targets: torch.Tensor,
        *,
        update_epochs: int,
        minibatch_size: int,
    ) -> PPOUpdateSummary:
        """Learn from one collected batch over several shuffled epochs.

        The batch holds ``N`` samples on the networks' device: ``observations``
        ``(N, observation_size)``; ``policy_actions`` ``(N,)`` for categorical
        or ``(N, action_size)`` for continuous PPO, as stored from
        ``PPOActionSample.policy_action``; and ``old_log_probabilities``,
        ``advantages``, and ``return_targets``, each ``(N,)``. Advantages are
        normalised once over the whole batch. Each epoch uses every sample
        once, in a new random order, in minibatches of ``minibatch_size``.
        """
        if update_epochs < 1:
            raise ValueError("Update epochs must be positive")
        if minibatch_size < 1:
            raise ValueError("Minibatch size must be positive")
        batch_size = observations.shape[0]
        if batch_size == 0 or any(
            values.shape[0] != batch_size
            for values in (
                policy_actions,
                old_log_probabilities,
                advantages,
                return_targets,
            )
        ):
            raise ValueError("Every input must hold the same nonzero number of samples")

        # Normalize once over the full batch, never independently per minibatch.
        # Nearly constant advantages are kept as they are rather than divided
        # by almost zero; torch.where makes that choice on the device.
        advantage_standard_deviation = advantages.std(correction=0)
        advantages = torch.where(
            advantage_standard_deviation > torch.finfo(advantages.dtype).eps,
            (advantages - advantages.mean()) / advantage_standard_deviation,
            advantages,
        )

        # The five measurements, added up over all minibatches on the device.
        measurement_totals = torch.zeros(5, device=self.device)
        minibatch_count = 0
        for _ in range(update_epochs):
            # Every epoch uses every transition once, in a new random order.
            shuffled_indices = torch.randperm(
                batch_size, generator=self.shuffle_generator, device=self.device
            )
            for minibatch_start in range(0, batch_size, minibatch_size):
                minibatch_indices = shuffled_indices[
                    minibatch_start : minibatch_start + minibatch_size
                ]
                measurement_totals += self._update_minibatch(
                    observations[minibatch_indices],
                    policy_actions[minibatch_indices],
                    old_log_probabilities[minibatch_indices],
                    advantages[minibatch_indices],
                    return_targets[minibatch_indices],
                )
                minibatch_count += 1

        # The only conversion to Python numbers: once, after all minibatches.
        return PPOUpdateSummary(*(measurement_totals / minibatch_count).tolist())

    def _update_minibatch(
        self,
        observations: torch.Tensor,
        policy_actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        return_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Take one clipped actor step and one critic step on a minibatch.

        Returns the five measurements, in the order of ``PPOUpdateSummary``'s
        fields, as one detached tensor of five values, so that ``update`` can
        add them up on the device without waiting for it.
        """

        # Re-evaluate the collected actions under the current, changing policy.
        new_log_probabilities, entropy_values = self.policy.evaluate_actions(
            observations,
            policy_actions,
        )

        log_probability_ratios = new_log_probabilities - old_log_probabilities
        probability_ratios = torch.exp(log_probability_ratios)
        clipped_probability_ratios = torch.clamp(
            probability_ratios,
            1 - self.clip_ratio,
            1 + self.clip_ratio,
        )

        unclipped_objectives = probability_ratios * advantages
        clipped_objectives = clipped_probability_ratios * advantages

        # The lower objective removes the reward for an excessively large change.
        surrogate_objectives = torch.minimum(
            unclipped_objectives,
            clipped_objectives,
        )

        mean_entropy = entropy_values.mean()
        approximate_kl = (probability_ratios - 1.0 - log_probability_ratios).mean()
        clip_fraction = (
            (torch.abs(probability_ratios - 1.0) > self.clip_ratio).float().mean()
        )
        actor_loss = (
            -surrogate_objectives.mean() - self.entropy_coefficient * mean_entropy
        )

        # Return targets were calculated before optimization and remain fixed.
        predicted_values = self.critic_network(observations)
        value_errors = return_targets - predicted_values
        critic_loss = 0.5 * value_errors.square().mean()

        # Actor and critic have independent networks and optimizers.
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        clip_gradients(self.actor_network.parameters(), self.max_gradient_norm)
        clip_gradients(self.critic_network.parameters(), self.max_gradient_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return torch.stack(
            (actor_loss, critic_loss, mean_entropy, approximate_kl, clip_fraction)
        ).detach()
