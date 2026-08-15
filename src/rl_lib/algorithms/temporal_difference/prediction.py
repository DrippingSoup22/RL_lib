"""One-step tabular temporal-difference prediction."""

import numpy as np


class TDZeroPrediction:
    """Estimate a fixed policy's state values with tabular TD(0)."""

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
        self.initial_value = initial_value

        self.V = np.full(number_of_states, self.initial_value, dtype=float)

    def update(
        self,
        state: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> None:

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite!")
        if not 0 <= state < self.number_of_states:
            raise ValueError("State must stay inside the states table!")
        if not 0 <= next_state < self.number_of_states:
            raise ValueError("Next state must stay inside the states table!")

        if terminated:
            target = reward
        else:
            target = reward + self.discount * self.V[next_state]

        error = target - self.V[state]
        self.V[state] += self.learning_rate * error
