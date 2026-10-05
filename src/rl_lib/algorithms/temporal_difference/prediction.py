"""Tabular temporal-difference prediction with rollout returns."""

from collections.abc import Sequence

import numpy as np

from rl_lib.trajectories import EpisodeStep, discounted_returns


class TDPrediction:
    """Estimate a fixed policy's state values from bootstrapped rollouts."""

    def __init__(
        self,
        number_of_states: int,
        learning_rate: float = 0.1,
        discount: float = 1.0,
        initial_value: float = 0.0,
    ) -> None:
        if number_of_states < 1:
            raise ValueError("Number of states must be at least 1")
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1]!")
        if not 0 <= discount <= 1:
            raise ValueError("Discount must be in [0, 1]!")
        if not np.isfinite(initial_value):
            raise ValueError("Initial value must be finite!")

        self.number_of_states = number_of_states
        self.learning_rate = learning_rate
        self.discount = discount
        self.V = np.full(number_of_states, initial_value, dtype=float)

    def update(
        self,
        steps: Sequence[EpisodeStep[int]],
        final_state: int,
        *,
        terminated: bool,
    ) -> tuple[float, ...]:
        """Update every state in a rollout and return its TD errors."""
        if not steps:
            raise ValueError("Steps must not be empty")
        if not 0 <= final_state < self.number_of_states:
            raise ValueError("Final state must stay inside the states table")
        for step in steps:
            if not 0 <= step.state < self.number_of_states:
                raise ValueError("State must stay inside the states table")
            if not np.isfinite(step.reward):
                raise ValueError("Reward must be finite")

        returns = discounted_returns(
            [step.reward for step in steps],
            self.discount,
            bootstrap=0.0 if terminated else float(self.V[final_state]),
        )

        errors = []
        for step, target in zip(steps, returns, strict=True):
            error = target - self.V[step.state]
            self.V[step.state] += self.learning_rate * error
            errors.append(float(error))
        return tuple(errors)
