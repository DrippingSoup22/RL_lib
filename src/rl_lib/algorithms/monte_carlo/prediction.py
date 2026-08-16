"""First-visit and every-visit Monte Carlo prediction."""

import numpy as np

from rl_lib.data.episode import Episode, discounted_returns


class FirstVisitMonteCarloPrediction:
    """Estimate a fixed policy's state values with first-visit returns."""

    def __init__(
        self,
        number_of_states: int,
        discount: float = 1.0,
    ) -> None:

        if number_of_states <= 0:
            raise ValueError("Number of state must be positive")
        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and between 0-1")
        self.number_of_states = number_of_states
        self.discount = discount
        self.visit_counts = np.zeros(number_of_states, dtype=int)
        self.V = np.zeros(number_of_states, dtype=float)

    def update(self, episode: Episode[int]) -> None:
        """Update estimates from one complete episode."""
        rewards = [step.reward for step in episode.steps]
        returns = discounted_returns(rewards, self.discount)

        visited_states = set()

        for step, episode_return in zip(episode.steps, returns, strict=True):
            state = step.state
            if not 0 <= state < self.number_of_states:
                raise ValueError("State value not valid!")
            if state not in visited_states:
                visited_states.add(state)
                self.visit_counts[state] += 1
                self.V[state] += (episode_return - self.V[state]) / self.visit_counts[
                    state
                ]


class EveryVisitMonteCarloPrediction:
    """Estimate a fixed policy's state values with every-visit returns."""

    def __init__(
        self,
        number_of_states: int,
        discount: float = 1.0,
    ) -> None:

        if number_of_states <= 0:
            raise ValueError("Number of state must be positive")
        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and between 0-1")
        self.number_of_states = number_of_states
        self.discount = discount
        self.visit_counts = np.zeros(number_of_states, dtype=int)
        self.V = np.zeros(number_of_states, dtype=float)

    def update(self, episode: Episode[int]) -> None:
        """Update estimates from one complete episode."""
        rewards = [step.reward for step in episode.steps]
        returns = discounted_returns(rewards, self.discount)

        for step, episode_return in zip(episode.steps, returns, strict=True):
            state = step.state
            if not 0 <= state < self.number_of_states:
                raise ValueError("State value not valid!")
            self.visit_counts[state] += 1
            self.V[state] += (episode_return - self.V[state]) / self.visit_counts[state]
