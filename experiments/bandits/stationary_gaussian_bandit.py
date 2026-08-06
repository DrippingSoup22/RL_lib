"""Stationary Gaussian k-armed bandit environment."""

from __future__ import annotations

import numpy as np


class StationaryGaussianBandit:
    def __init__(
        self, k: int = 10, reward_std: float = 1.0, seed: int | None = None
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        if reward_std < 0:
            raise ValueError("reward_std must be non-negative")

        self.k = k
        self.reward_std = reward_std
        self.rng = np.random.default_rng(seed)

        self.action_values = self.rng.normal(loc=0.0, scale=1.0, size=k)
        self.best_action = int(np.argmax(self.action_values))
        self.best_value = self.action_values[self.best_action]

    @property
    def optimal_action(self) -> int:
        """Return the arm with the greatest true value."""
        return self.best_action

    def step(self, action: int) -> float:
        """Select an arm and sample its reward."""
        if action < 0 or action >= self.k:
            raise ValueError(f"action must be between 0 and {self.k - 1}")

        reward = self.rng.normal(loc=self.action_values[action], scale=self.reward_std)
        return float(reward)
