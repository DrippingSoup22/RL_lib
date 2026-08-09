"""Tabular policies and policy-improvement operations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def epsilon_soft_probabilities(
    action_values: ArrayLike,
    epsilon: float,
) -> NDArray[np.float64]:
    """Calculate epsilon-soft probabilities for a given set of action values."""
    values = np.asarray(action_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("action_values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("action_values must be finite")
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and between 0 and 1")

    probabilities = np.full(values.size, epsilon / values.size)
    max_indices = np.flatnonzero(values == np.max(values))
    probabilities[max_indices] += (1.0 - epsilon) / max_indices.size

    return probabilities


class TabularPolicy:
    """A stochastic policy for finite discrete state and action spaces."""

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        seed: int | None = None,
    ) -> None:
        if number_of_states < 1:
            raise ValueError("number_of_states must be at least 1")
        if number_of_actions < 1:
            raise ValueError("number_of_actions must be at least 1")

        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.rng = np.random.default_rng(seed)
        self.probabilities = np.full(
            (number_of_states, number_of_actions),
            1.0 / number_of_actions,
        )

    def select_action(self, state: int) -> int:
        """Sample an action from the probability distribution for ``state``."""
        self._validate_state(state)
        return int(
            self.rng.choice(
                self.number_of_actions,
                p=self.probabilities[state],
            )
        )

    def set_action_probabilities(
        self,
        state: int,
        probabilities: ArrayLike,
    ) -> None:
        """Replace the complete action distribution for one state."""
        self._validate_state(state)
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (self.number_of_actions,):
            raise ValueError("probabilities must contain one value for every action")
        if not np.all(np.isfinite(values)):
            raise ValueError("probabilities must be finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("probabilities must be between 0 and 1")
        if not np.isclose(np.sum(values), 1.0):
            raise ValueError("probabilities must sum to 1")

        self.probabilities[state] = values

    def _validate_state(self, state: int) -> None:
        if state < 0 or state >= self.number_of_states:
            raise ValueError(f"state must be between 0 and {self.number_of_states - 1}")


def policy_from_action_values(
    action_values: ArrayLike,
    epsilon: float,
    seed: int | None = None,
) -> TabularPolicy:
    values = np.asarray(action_values, dtype=float)

    if not values.ndim == 2:
        raise ValueError("Dim values is different than 2!")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("States or Actions must be minimum of 1 each!")
    if not np.all(np.isfinite(values)):
        raise ValueError("All value must be finite!")

    number_of_states, number_of_actions = values.shape

    policy = TabularPolicy(
        number_of_states,
        number_of_actions,
        seed,
    )

    for state in range(number_of_states):
        probabilities = epsilon_soft_probabilities(values[state], epsilon)
        policy.set_action_probabilities(state, probabilities)

    return policy
