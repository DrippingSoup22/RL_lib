"""Non-stationary Gaussian k-armed bandit environment."""

from __future__ import annotations

import numpy as np


class NonstationaryGaussianBandit:
    def __init__(
        self,
        k: int = 10,
        reward_std: float = 1.0,
        drift_std: float = 0.01,
        initial_value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        if reward_std < 0:
            raise ValueError("reward_std must be non-negative")
        if drift_std < 0:
            raise ValueError("drift_std must be non-negative")
        if not np.isfinite(initial_value):
            raise ValueError("initial_value must be finite")

        self.k = k
        self.reward_std = reward_std
        self.drift_std = drift_std
        self.rng = np.random.default_rng(seed)

        self.action_values = np.full(
            shape=k,
            fill_value=initial_value,
            dtype=float,
        )

    @property
    def optimal_action(self) -> int:
        """Return the arm with the greatest current true value."""
        return int(np.argmax(self.action_values))

    def step(self, action: int) -> float:
        if action < 0 or action >= self.k:
            raise ValueError(f"action must be between 0 and {self.k - 1}")
        reward = self.rng.normal(
            loc=self.action_values[action],
            scale=self.reward_std,
        )
        self.action_values += self.rng.normal(
            loc=0,
            scale=self.drift_std,
            size=self.k,
        )
        return float(reward)
