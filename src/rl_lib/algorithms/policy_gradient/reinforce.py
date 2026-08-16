"""Monte Carlo policy-gradient control for discrete actions."""

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from rl_lib.data import Episode, discounted_returns
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork


class Reinforce:
    """Episodic REINFORCE with a stochastic discrete policy."""

    def __init__(
        self,
        actor_model: DiscretePolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
    ) -> None:

        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and in [0, 1]!")

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.discount = discount

    def select_action(self, observation: ArrayLike) -> int:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

        with torch.no_grad():
            logits = self.actor_model(observation_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
        return int(action.item())

    def _episode_tensors(
        self,
        episode: Episode[NDArray[np.float32]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not episode.steps:
            raise ValueError("Episode must contain at least one step")

        observations = np.asarray(
            [step.state for step in episode.steps],
            dtype=np.float32,
        )
        if observations.shape != (
            len(episode.steps),
            self.actor_model.observation_size,
        ):
            raise ValueError("Episode observations must match the model input size")
        if not np.all(np.isfinite(observations)):
            raise ValueError("Episode observations must be finite")

        if any(
            isinstance(step.action, (bool, np.bool_))
            or not isinstance(step.action, (int, np.integer))
            for step in episode.steps
        ):
            raise ValueError("Episode actions must be integers")
        actions = np.asarray(
            [step.action for step in episode.steps],
            dtype=np.int64,
        )
        if np.any((actions < 0) | (actions >= self.actor_model.number_of_actions)):
            raise ValueError("Episode actions must stay inside the action space")

        rewards = np.asarray(
            [step.reward for step in episode.steps],
            dtype=float,
        )
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Episode rewards must be finite")

        returns = discounted_returns(rewards, self.discount)

        observations_tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
        )
        actions_tensor = torch.as_tensor(
            actions,
            dtype=torch.int64,
        )
        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
        )
        return observations_tensor, actions_tensor, returns_tensor

    def update(self, episode: Episode[NDArray[np.float32]]) -> float:

        observations_tensor, actions_tensor, returns_tensor = self._episode_tensors(
            episode
        )

        logits = self.actor_model(observations_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(actions_tensor)

        discount_weights = self.discount ** torch.arange(
            len(returns_tensor),
            dtype=torch.float32,
        )

        loss = -(discount_weights * returns_tensor * log_probabilities).sum()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return float(loss.item())


class ReinforceWithBaseline(Reinforce):
    """REINFORCE using a learned state-value baseline."""

    def __init__(
        self,
        actor_model: DiscretePolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        critic_model: StateValueNetwork,
        critic_optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
    ) -> None:

        super().__init__(actor_model, actor_optimizer, discount)

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

        observations_tensor, actions_tensor, returns_tensor = self._episode_tensors(
            episode
        )

        logits = self.actor_model(observations_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(actions_tensor)

        values = self.critic_model(observations_tensor)
        advantages = returns_tensor - values.detach()

        discount_weights = self.discount ** torch.arange(
            len(episode.steps), dtype=torch.float32
        )

        actor_loss = -(discount_weights * advantages * log_probabilities).sum()

        critic_error = returns_tensor - values
        critic_loss = 0.5 * critic_error.square().sum()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return float(actor_loss.item())
