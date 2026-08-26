"""Semi-gradient TD prediction with bootstrapped rollout returns."""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from rl_lib.data import EpisodeStep, rollout_arrays
from rl_lib.models import StateValueNetwork


class SemiGradientTDPrediction:
    """Estimate a fixed policy's value function from rollout returns."""

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
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
        *,
        terminated: bool,
    ) -> tuple[float, ...]:
        """Take one semi-gradient update over a complete rollout."""
        arrays = rollout_arrays(
            steps,
            final_state,
            observation_size=self.model.observation_size,
        )
        observations_tensor = torch.as_tensor(arrays.observations)
        rewards_tensor = torch.as_tensor(arrays.rewards)
        final_state_tensor = torch.as_tensor(arrays.final_state)

        returns = torch.empty_like(rewards_tensor)
        with torch.no_grad():
            running_return = (
                rewards_tensor.new_zeros(())
                if terminated
                else self.model(final_state_tensor)
            )
            for index in range(len(steps) - 1, -1, -1):
                running_return = rewards_tensor[index] + self.discount * running_return
                returns[index] = running_return

        errors = returns - self.model(observations_tensor)
        loss = 0.5 * errors.square().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return tuple(float(error) for error in errors.detach().tolist())
