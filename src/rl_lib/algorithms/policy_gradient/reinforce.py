"""Monte Carlo policy-gradient control."""

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from rl_lib.data import Episode, discounted_returns, rollout_arrays
from rl_lib.models import (
    DiscretePolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)
from rl_lib.optimizers import clip_gradients, validate_max_gradient_norm
from rl_lib.policies import CategoricalPolicy, SquashedGaussianPolicy


class Reinforce:
    """Episodic REINFORCE with a categorical or continuous policy."""

    def __init__(
        self,
        actor_model: DiscretePolicyNetwork | GaussianPolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
        *,
        max_gradient_norm: float | None = None,
        action_low: ArrayLike | None = None,
        action_high: ArrayLike | None = None,
    ) -> None:

        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and in [0, 1]!")

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.discount = discount
        self.max_gradient_norm = validate_max_gradient_norm(max_gradient_norm)
        self.policy: CategoricalPolicy | SquashedGaussianPolicy

        if isinstance(actor_model, DiscretePolicyNetwork):
            if action_low is not None or action_high is not None:
                raise ValueError(
                    "Categorical REINFORCE must not receive continuous action bounds"
                )
            self.policy = CategoricalPolicy(actor_model)
        elif isinstance(actor_model, GaussianPolicyNetwork):
            if action_low is None or action_high is None:
                raise ValueError(
                    "Continuous REINFORCE requires lower and upper action bounds"
                )
            self.policy = SquashedGaussianPolicy(actor_model, action_low, action_high)
        else:
            raise TypeError("Actor model must be discrete or Gaussian")

    def sample_action(
        self,
        observation: ArrayLike,
    ) -> tuple[int, int] | tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return environment and policy actions from one policy sample."""
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

        with torch.no_grad():
            if isinstance(self.policy, CategoricalPolicy):
                action, _ = self.policy.sample(observation_tensor)
                action_index = int(action.item())
                return action_index, action_index

            # Keep the unsquashed action for learning; only its bounded
            # transformation is sent to the continuous environment.
            environment_action, latent_action, _ = self.policy.sample(
                observation_tensor
            )
            return (
                environment_action.cpu().numpy().astype(np.float32, copy=True),
                latent_action.cpu().numpy().astype(np.float32, copy=True),
            )

    def select_action(
        self,
        observation: ArrayLike,
        *,
        deterministic: bool = False,
    ) -> int | NDArray[np.float32]:
        if not deterministic:
            environment_action, _ = self.sample_action(observation)
            return environment_action

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action = self.policy.deterministic_action(observation_tensor)
        if isinstance(self.policy, CategoricalPolicy):
            return int(action.item())
        return action.cpu().numpy().astype(np.float32, copy=True)

    def _episode_tensors(
        self,
        episode: Episode[NDArray[np.float32]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Categorical policies reevaluate indices; continuous policies
        # reevaluate the unsquashed Gaussian samples stored during interaction.
        if isinstance(self.policy, CategoricalPolicy):
            arrays = rollout_arrays(
                episode.steps,
                episode.final_state,
                observation_size=self.actor_model.observation_size,
                number_of_actions=self.policy.model.number_of_actions,
            )
        else:
            arrays = rollout_arrays(
                episode.steps,
                episode.final_state,
                observation_size=self.actor_model.observation_size,
                action_size=self.policy.model.action_size,
            )
        returns = discounted_returns(arrays.rewards, self.discount)
        return (
            torch.as_tensor(arrays.observations),
            torch.as_tensor(arrays.actions),
            torch.as_tensor(returns, dtype=torch.float32),
        )

    def update(self, episode: Episode[NDArray[np.float32]]) -> float:

        observations_tensor, policy_actions_tensor, returns_tensor = (
            self._episode_tensors(episode)
        )

        log_probabilities, _ = self.policy.evaluate_actions(
            observations_tensor,
            policy_actions_tensor,
        )

        discount_weights = self.discount ** torch.arange(
            len(returns_tensor),
            dtype=torch.float32,
        )

        # Each return supplies the direction and magnitude for increasing or
        # decreasing the log-probability of the action taken at that step.
        loss = -(discount_weights * returns_tensor * log_probabilities).sum()

        self.actor_optimizer.zero_grad()
        loss.backward()
        clip_gradients(self.actor_model.parameters(), self.max_gradient_norm)
        self.actor_optimizer.step()

        return float(loss.item())


class ReinforceWithBaseline(Reinforce):
    """REINFORCE using a learned state-value baseline."""

    def __init__(
        self,
        actor_model: DiscretePolicyNetwork | GaussianPolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        critic_model: StateValueNetwork,
        critic_optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
        *,
        max_gradient_norm: float | None = None,
        action_low: ArrayLike | None = None,
        action_high: ArrayLike | None = None,
    ) -> None:

        super().__init__(
            actor_model,
            actor_optimizer,
            discount,
            max_gradient_norm=max_gradient_norm,
            action_low=action_low,
            action_high=action_high,
        )

        if actor_model.observation_size != critic_model.observation_size:
            raise ValueError(
                "Policy and value models must use the same observation size"
            )
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer

    def update(
        self,
        episode: Episode[NDArray[np.float32]],
    ) -> float:

        observations_tensor, policy_actions_tensor, returns_tensor = (
            self._episode_tensors(episode)
        )

        log_probabilities, _ = self.policy.evaluate_actions(
            observations_tensor,
            policy_actions_tensor,
        )

        values = self.critic_model(observations_tensor)
        # The baseline reduces variance but is held fixed during the actor step.
        advantages = returns_tensor - values.detach()

        discount_weights = self.discount ** torch.arange(
            len(episode.steps), dtype=torch.float32
        )

        actor_loss = -(discount_weights * advantages * log_probabilities).sum()

        critic_error = returns_tensor - values
        # The critic independently regresses toward the Monte Carlo returns.
        critic_loss = 0.5 * critic_error.square().sum()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        clip_gradients(self.actor_model.parameters(), self.max_gradient_norm)
        clip_gradients(self.critic_model.parameters(), self.max_gradient_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return float(actor_loss.item())
