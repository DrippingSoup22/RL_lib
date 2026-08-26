"""Clipped proximal policy optimization for discrete actions."""

import numpy as np
import torch
from numpy.typing import ArrayLike

from rl_lib.data import PPOActionSample, PPOUpdateResult
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork


class PPO:
    """A discrete-action PPO actor and state-value critic."""

    def __init__(
        self,
        actor_model: DiscretePolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        critic_model: StateValueNetwork,
        critic_optimizer: torch.optim.Optimizer,
        clip_ratio: float = 0.2,
        entropy_coefficient: float = 0.0,
        seed: int | None = None,
    ) -> None:

        if not actor_model.observation_size == critic_model.observation_size:
            raise ValueError("Models observation sizes must coincide!")
        if not np.isfinite(clip_ratio) or not 0 < clip_ratio < 1:
            raise ValueError("Clip ratio must be finite and in (0, 1)!")
        if not np.isfinite(entropy_coefficient) or entropy_coefficient < 0:
            raise ValueError("Entropy coefficient must be finite and nonnegative!")

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.clip_ratio = clip_ratio
        self.entropy_coefficient = entropy_coefficient
        self.rng = np.random.default_rng(seed)

    def sample_action(self, observation: ArrayLike) -> PPOActionSample:
        """Sample once and retain the behavior policy measurements PPO will freeze."""
        observation_tensor = self._observation_tensor(observation)

        with torch.no_grad():
            logits = self.actor_model(observation_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)

            value = self.critic_model(observation_tensor)

        return PPOActionSample(
            int(action.item()), float(log_probability.item()), float(value.item())
        )

    def select_action(self, observation: ArrayLike) -> int:

        observation_tensor = self._observation_tensor(observation)

        with torch.no_grad():
            logits = self.actor_model(observation_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
        return int(action.item())

    def state_value(self, observation: ArrayLike) -> float:
        observation_tensor = self._observation_tensor(observation)

        with torch.no_grad():
            value = self.critic_model(observation_tensor)
        return float(value.item())

    def update(
        self,
        observations: ArrayLike,
        actions: ArrayLike,
        old_log_probabilities: ArrayLike,
        advantages: ArrayLike,
        return_targets: ArrayLike,
        *,
        update_epochs: int,
        minibatch_size: int,
    ) -> tuple[PPOUpdateResult, ...]:
        """Reuse one frozen batch over shuffled PPO epochs."""
        if update_epochs < 1:
            raise ValueError("Update epochs must be positive")
        if minibatch_size < 1:
            raise ValueError("Minibatch size must be positive")

        (
            observations_tensor,
            actions_tensor,
            old_log_probabilities_tensor,
            advantages_tensor,
            return_targets_tensor,
        ) = self._minibatch_tensors(
            observations,
            actions,
            old_log_probabilities,
            advantages,
            return_targets,
        )

        # These behavior-policy measurements stay fixed while the models change.
        observations_array = observations_tensor.numpy()
        actions_array = actions_tensor.numpy()
        old_log_probabilities_array = old_log_probabilities_tensor.numpy()
        advantages_array = advantages_tensor.numpy()
        return_targets_array = return_targets_tensor.numpy()

        # Normalize once over the full batch, never independently per minibatch.
        advantage_standard_deviation = float(np.std(advantages_array))
        if advantage_standard_deviation > np.finfo(np.float32).eps:
            advantages_array = (
                advantages_array - np.mean(advantages_array)
            ) / advantage_standard_deviation

        update_results: list[PPOUpdateResult] = []
        batch_size = len(observations_array)
        for _ in range(update_epochs):
            # Every epoch uses every transition once, in a new random order.
            shuffled_indices = self.rng.permutation(batch_size)
            for minibatch_start in range(0, batch_size, minibatch_size):
                minibatch_indices = shuffled_indices[
                    minibatch_start : minibatch_start + minibatch_size
                ]
                update_results.append(
                    self.update_minibatch(
                        observations_array[minibatch_indices],
                        actions_array[minibatch_indices],
                        old_log_probabilities_array[minibatch_indices],
                        advantages_array[minibatch_indices],
                        return_targets_array[minibatch_indices],
                    )
                )
        return tuple(update_results)

    def update_minibatch(
        self,
        observations: ArrayLike,
        actions: ArrayLike,
        old_log_probabilities: ArrayLike,
        advantages: ArrayLike,
        return_targets: ArrayLike,
    ) -> PPOUpdateResult:
        """Take one clipped actor step and one critic regression step."""
        (
            observations_tensor,
            actions_tensor,
            old_log_probabilities_tensor,
            advantages_tensor,
            return_targets_tensor,
        ) = self._minibatch_tensors(
            observations,
            actions,
            old_log_probabilities,
            advantages,
            return_targets,
        )

        # Re-evaluate the collected actions under the current, changing policy.
        logits = self.actor_model(observations_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_probabilities = distribution.log_prob(actions_tensor)

        probability_ratios = torch.exp(
            new_log_probabilities - old_log_probabilities_tensor
        )
        clipped_probability_ratios = torch.clamp(
            probability_ratios,
            1 - self.clip_ratio,
            1 + self.clip_ratio,
        )

        unclipped_objectives = probability_ratios * advantages_tensor
        clipped_objectives = clipped_probability_ratios * advantages_tensor

        # The lower objective removes the reward for an excessively large change.
        surrogate_objectives = torch.minimum(
            unclipped_objectives,
            clipped_objectives,
        )

        entropy = distribution.entropy().mean()
        actor_loss = -surrogate_objectives.mean() - self.entropy_coefficient * entropy

        # Return targets were calculated before optimization and remain fixed.
        predicted_values = self.critic_model(observations_tensor)
        value_errors = return_targets_tensor - predicted_values
        critic_loss = 0.5 * value_errors.square().mean()

        # Actor and critic have independent models and optimizers.
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return PPOUpdateResult(
            float(actor_loss.item()), float(critic_loss.item()), float(entropy.item())
        )

    def _minibatch_tensors(
        self,
        observations: ArrayLike,
        actions: ArrayLike,
        old_log_probabilities: ArrayLike,
        advantages: ArrayLike,
        return_targets: ArrayLike,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        observations_array = np.asarray(observations, dtype=np.float32)
        actions_array = np.asarray(actions)
        old_log_probabilities_array = np.asarray(
            old_log_probabilities,
            dtype=np.float32,
        )
        advantages_array = np.asarray(advantages, dtype=np.float32)
        return_targets_array = np.asarray(return_targets, dtype=np.float32)

        expected_observation_size = self.actor_model.observation_size
        if (
            observations_array.ndim != 2
            or observations_array.shape[1] != expected_observation_size
        ):
            raise ValueError(
                f"Observations must have shape (batch, {expected_observation_size})"
            )
        batch_size = observations_array.shape[0]
        if batch_size == 0:
            raise ValueError("PPO minibatches must be nonempty")
        if not np.all(np.isfinite(observations_array)):
            raise ValueError("All observation values must be finite")

        if actions_array.shape != (batch_size,):
            raise ValueError("Actions must contain one value per observation")
        # Validate before int64 conversion, which would silently truncate floats.
        if not np.issubdtype(actions_array.dtype, np.integer):
            raise ValueError("Actions must be integers")
        if np.any(
            (actions_array < 0) | (actions_array >= self.actor_model.number_of_actions)
        ):
            raise ValueError("Actions must stay inside the action space")

        vector_fields = (
            ("Old log-probabilities", old_log_probabilities_array),
            ("Advantages", advantages_array),
            ("Return targets", return_targets_array),
        )
        for name, field_values in vector_fields:
            if field_values.shape != (batch_size,):
                raise ValueError(f"{name} must contain one value per observation")
            if not np.all(np.isfinite(field_values)):
                raise ValueError(f"All {name.lower()} must be finite")

        return (
            torch.as_tensor(observations_array, dtype=torch.float32),
            torch.as_tensor(actions_array, dtype=torch.int64),
            torch.as_tensor(old_log_probabilities_array, dtype=torch.float32),
            torch.as_tensor(advantages_array, dtype=torch.float32),
            torch.as_tensor(return_targets_array, dtype=torch.float32),
        )

    def _observation_tensor(self, observation: ArrayLike) -> torch.Tensor:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        expected_shape = (self.actor_model.observation_size,)

        if observation_tensor.shape != expected_shape:
            raise ValueError(
                f"Observation must have shape {expected_shape}, "
                f"received {tuple(observation_tensor.shape)}"
            )
        if not torch.isfinite(observation_tensor).all():
            raise ValueError("All observation values must be finite!")

        return observation_tensor
