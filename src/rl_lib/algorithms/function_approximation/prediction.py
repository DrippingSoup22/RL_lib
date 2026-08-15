"""Semi-gradient TD prediction with a differentiable value model."""

import numpy as np
import torch
from numpy.typing import ArrayLike

from rl_lib.models import StateValueNetwork


class SemiGradientTDZeroPrediction:
    """Estimate a fixed policy's value function with semi-gradient TD(0)."""

    def __init__(
        self,
        model: StateValueNetwork,
        optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
    ) -> None:

        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and in [0, 1]")

        self.model = model
        self.optimizer = optimizer
        self.discount = discount

    def update(
        self,
        observation: ArrayLike,
        reward: float,
        next_observation: ArrayLike,
        terminated: bool,
    ) -> float:

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        next_observation_tensor = torch.as_tensor(next_observation, dtype=torch.float32)

        with torch.no_grad():
            if terminated:
                target = torch.as_tensor(reward, dtype=torch.float32)
            else:
                target = reward + self.discount * self.model(next_observation_tensor)

        error = target - self.model(observation_tensor)
        loss = 0.5 * error.square()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return error.item()
