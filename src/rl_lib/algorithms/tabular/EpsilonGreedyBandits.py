"""Epsilon-greedy algorithm for k-armed bandits."""

from __future__ import annotations

import numpy as np


class EpsilonGreedyBandits:
    def __init__(
        self,
        k: int,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        self.k = k
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.rng = np.random.default_rng(seed)

        self.estimates = np.full(k, initial_value, dtype=float)
        self.counts = np.zeros(k, dtype=int)

    def select_action(self) -> int:
        """Select an action using an epsilon-greedy policy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        best_estimate = np.max(self.estimates)
        greedy_actions = np.flatnonzero(self.estimates == best_estimate)
        return int(self.rng.choice(greedy_actions))

    def update(self, action: int, reward: float) -> None:
        """Update the selected action's value estimate."""
        if action < 0 or action >= self.k:
            raise ValueError(f"action must be between 0 and {self.k - 1}")
        self.counts[action] += 1
        self.estimates[action] += (reward - self.estimates[action]) / self.counts[
            action
        ]
