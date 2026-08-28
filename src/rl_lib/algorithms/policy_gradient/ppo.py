"""Clipped proximal policy optimization."""

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from rl_lib.data import (
    CategoricalPPOActionSample,
    ContinuousPPOActionSample,
    PPOUpdateResult,
)
from rl_lib.models import (
    DiscretePolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)
from rl_lib.optimizers import clip_gradients, validate_max_gradient_norm
from rl_lib.policies import CategoricalPolicy, SquashedGaussianPolicy


class PPO:
    """PPO actor and critic for categorical or bounded continuous actions."""

    def __init__(
        self,
        actor_model: DiscretePolicyNetwork | GaussianPolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        critic_model: StateValueNetwork,
        critic_optimizer: torch.optim.Optimizer,
        clip_ratio: float = 0.2,
        entropy_coefficient: float = 0.0,
        seed: int | None = None,
        *,
        max_gradient_norm: float | None = None,
        action_low: ArrayLike | None = None,
        action_high: ArrayLike | None = None,
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
        self.max_gradient_norm = validate_max_gradient_norm(max_gradient_norm)
        self.rng = np.random.default_rng(seed)
        self.policy: CategoricalPolicy | SquashedGaussianPolicy

        if isinstance(actor_model, DiscretePolicyNetwork):
            if action_low is not None or action_high is not None:
                raise ValueError(
                    "Categorical PPO must not receive continuous action bounds"
                )
            self.policy = CategoricalPolicy(actor_model)

        elif isinstance(actor_model, GaussianPolicyNetwork):
            if action_low is None or action_high is None:
                raise ValueError(
                    "Continuous PPO requires lower and upper action bounds"
                )
            self.policy = SquashedGaussianPolicy(actor_model, action_low, action_high)

        else:
            raise TypeError("Actor model must be discrete or Gaussian")

    def sample_action(
        self, observation: ArrayLike
    ) -> CategoricalPPOActionSample | ContinuousPPOActionSample:
        """Sample once and freeze the behavior-policy measurements."""

        observation_tensor = self._observation_tensor(observation)
        with torch.no_grad():
            value = self.critic_model(observation_tensor)

            if isinstance(self.policy, CategoricalPolicy):
                action, log_probability = self.policy.sample(observation_tensor)

                return CategoricalPPOActionSample(
                    action=int(action.item()),
                    log_probability=float(log_probability.item()),
                    value=float(value.item()),
                )

            environment_action, latent_action, log_probability = self.policy.sample(
                observation_tensor
            )
            return ContinuousPPOActionSample(
                action=environment_action.cpu().numpy().astype(np.float32, copy=True),
                latent_action=latent_action.cpu().numpy().astype(np.float32, copy=True),
                log_probability=float(log_probability.item()),
                value=float(value.item()),
            )

    def select_action(
        self,
        observation: ArrayLike,
        *,
        deterministic: bool = False,
    ) -> int | NDArray[np.float32]:
        observation_tensor = self._observation_tensor(observation)

        with torch.no_grad():
            if deterministic:
                action = self.policy.deterministic_action(observation_tensor)
                if isinstance(self.policy, CategoricalPolicy):
                    return int(action.item())
                return action.cpu().numpy().astype(np.float32, copy=True)
            if isinstance(self.policy, CategoricalPolicy):
                action, _ = self.policy.sample(observation_tensor)
                return int(action.item())

            environment_action, _, _ = self.policy.sample(observation_tensor)
            return environment_action.cpu().numpy().astype(np.float32, copy=True)

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
        new_log_probabilities, entropy_values = self.policy.evaluate_actions(
            observations_tensor,
            actions_tensor,
        )

        log_probability_ratios = new_log_probabilities - old_log_probabilities_tensor
        probability_ratios = torch.exp(log_probability_ratios)
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

        entropy = entropy_values.mean()
        approximate_kl = (probability_ratios - 1.0 - log_probability_ratios).mean()
        clip_fraction = (
            (torch.abs(probability_ratios - 1.0) > self.clip_ratio).float().mean()
        )
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
        clip_gradients(self.actor_model.parameters(), self.max_gradient_norm)
        clip_gradients(self.critic_model.parameters(), self.max_gradient_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return PPOUpdateResult(
            float(actor_loss.item()),
            float(critic_loss.item()),
            float(entropy.item()),
            float(approximate_kl.item()),
            float(clip_fraction.item()),
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

        if isinstance(self.policy, CategoricalPolicy):
            actions_array = np.asarray(actions)
            if actions_array.shape != (batch_size,):
                raise ValueError("Actions must contain one value per observation")
            # Validate before int64 conversion, which would truncate floats.
            if not np.issubdtype(actions_array.dtype, np.integer):
                raise ValueError("Actions must be integers")
            if np.any(
                (actions_array < 0)
                | (actions_array >= self.policy.model.number_of_actions)
            ):
                raise ValueError("Actions must stay inside the action space")
            actions_tensor = torch.as_tensor(actions_array, dtype=torch.int64)
        else:
            try:
                actions_array = np.asarray(actions, dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError("Latent actions must be numeric") from error
            expected_action_shape = (batch_size, self.policy.model.action_size)
            if actions_array.shape != expected_action_shape:
                raise ValueError(
                    f"Latent actions must have shape {expected_action_shape}"
                )
            if not np.all(np.isfinite(actions_array)):
                raise ValueError("All latent action values must be finite")
            actions_tensor = torch.as_tensor(actions_array, dtype=torch.float32)

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
            actions_tensor,
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
