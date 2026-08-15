"""Semi-gradient control for continuous observations and discrete actions."""

import numpy as np
import torch
from numpy.typing import ArrayLike

from rl_lib.models import ActionValueNetwork
from rl_lib.policies import epsilon_soft_probabilities


class SemiGradientSARSA:
    """On-policy semi-gradient SARSA with epsilon-greedy action selection."""

    def __init__(
        self,
        model: ActionValueNetwork,
        optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:

        if not 0 <= discount <= 1:
            raise ValueError("Discount must be in [0, 1]!")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in [0, 1]!")

        self.model = model
        self.optimizer = optimizer
        self.discount = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: ArrayLike) -> int:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

        with torch.no_grad():
            action_values = self.model(observation_tensor)

        probabilities = epsilon_soft_probabilities(action_values.numpy(), self.epsilon)
        return int(self.rng.choice(self.model.number_of_actions, p=probabilities))

    def update(
        self,
        observation: ArrayLike,
        action: int,
        reward: float,
        next_observation: ArrayLike,
        next_action: int | None,
        terminated: bool,
    ) -> float:

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        if not 0 <= action < self.model.number_of_actions:
            raise ValueError("Action must stay inside the possible actions!")

        if not terminated:
            if next_action is None:
                raise ValueError(
                    "next_action is required for a non-terminal transition"
                )
            if not 0 <= next_action < self.model.number_of_actions:
                raise ValueError("Next action must stay inside the possible actions!")

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        next_observation_tensor = torch.as_tensor(next_observation, dtype=torch.float32)

        with torch.no_grad():
            if terminated:
                target = torch.as_tensor(reward, dtype=torch.float32)
            else:
                target = (
                    reward
                    + self.discount * self.model(next_observation_tensor)[next_action]
                )

        error = target - self.model(observation_tensor)[action]
        loss = 0.5 * error.square()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return error.item()


class SemiGradientQLearning:
    """Off-policy semi-gradient Q-learning with epsilon-greedy behavior."""

    def __init__(
        self,
        model: ActionValueNetwork,
        optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:

        if not 0 <= discount <= 1:
            raise ValueError("Discount must be in [0, 1]!")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in [0, 1]!")

        self.model = model
        self.optimizer = optimizer
        self.discount = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: ArrayLike) -> int:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)

        with torch.no_grad():
            action_values = self.model(observation_tensor)

        probabilities = epsilon_soft_probabilities(action_values.numpy(), self.epsilon)
        return int(self.rng.choice(self.model.number_of_actions, p=probabilities))

    def update(
        self,
        observation: ArrayLike,
        action: int,
        reward: float,
        next_observation: ArrayLike,
        terminated: bool,
    ) -> float:

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        if not 0 <= action < self.model.number_of_actions:
            raise ValueError("Action must stay inside the possible actions!")

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        next_observation_tensor = torch.as_tensor(next_observation, dtype=torch.float32)

        with torch.no_grad():
            if terminated:
                target = torch.as_tensor(reward, dtype=torch.float32)
            else:
                next_action_values = self.model(next_observation_tensor)
                best_next_value = torch.max(next_action_values)
                target = reward + self.discount * best_next_value

        error = target - self.model(observation_tensor)[action]
        loss = 0.5 * error.square()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return error.item()
