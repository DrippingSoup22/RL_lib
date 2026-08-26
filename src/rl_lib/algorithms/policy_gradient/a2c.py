"""Synchronous advantage actor-critic for discrete actions."""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from rl_lib.data import EpisodeStep, rollout_arrays
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork


class A2C:
    def __init__(
        self,
        actor_model: DiscretePolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        critic_model: StateValueNetwork,
        critic_optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
        entropy_coefficient: float = 0.0,
    ) -> None:

        if not actor_model.observation_size == critic_model.observation_size:
            raise ValueError("Models must have the same observation size!")
        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and in [0, 1]!")
        if not np.isfinite(entropy_coefficient) or not entropy_coefficient >= 0:
            raise ValueError("Entropy coefficient must be finite and non negative!")

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.discount = discount
        self.entropy_coefficient = entropy_coefficient

    def select_action(self, observation: ArrayLike) -> int:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
        )

        with torch.no_grad():
            logits = self.actor_model(observation_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
        return int(action.item())

    def _rollout_tensors(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        arrays = rollout_arrays(
            steps,
            final_state,
            observation_size=self.actor_model.observation_size,
            number_of_actions=self.actor_model.number_of_actions,
        )
        return (
            torch.as_tensor(arrays.observations),
            torch.as_tensor(arrays.actions),
            torch.as_tensor(arrays.rewards),
            torch.as_tensor(arrays.final_state),
        )

    def update(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
        *,
        terminated: bool,
    ) -> tuple[float, float]:

        observations_tensor, actions_tensor, rewards_tensor, final_state_tensor = (
            self._rollout_tensors(steps, final_state)
        )

        returns = torch.empty_like(rewards_tensor)
        with torch.no_grad():
            if terminated:
                running_return = rewards_tensor.new_zeros(())
            else:
                running_return = self.critic_model(final_state_tensor)

            for index in range(len(rewards_tensor) - 1, -1, -1):
                running_return = rewards_tensor[index] + self.discount * running_return
                returns[index] = running_return

        logits = self.actor_model(observations_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(actions_tensor)
        entropy = distribution.entropy()

        values = self.critic_model(observations_tensor)
        advantages = returns - values

        actor_loss = -(log_probabilities * advantages.detach()).mean()
        actor_loss -= self.entropy_coefficient * entropy.mean()

        critic_loss = 0.5 * advantages.square().mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return float(actor_loss.item()), float(critic_loss.item())
