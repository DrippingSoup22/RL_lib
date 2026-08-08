"""Greedy policy bandits with UCB"""

from __future__ import annotations

import numpy as np


class UCBGreedyBandits:
    def __init__(
        self,
        k: int,
        c: float = 1.0,
        initial_value: float = 0.0,
        seed: int | None = None,
        step_size: float | None = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        if not np.isfinite(c) or c < 0:
            raise ValueError("c must be finite and non-negative")
        if step_size is not None and not 0 < step_size <= 1:
            raise ValueError("step_size must be between 0 exclusive and 1 inclusive")

        self.k = k
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.step_size = step_size

        self.estimates = np.full(k, initial_value, dtype=float)
        self.counts = np.zeros(k, dtype=int)

    @property
    def values(self) -> np.ndarray:
        """Backward-compatible alias for the action-value estimates."""
        return self.estimates

    def select_action(self) -> int:
        untried_actions = np.flatnonzero(self.counts == 0)
        if len(untried_actions) > 0:
            return int(self.rng.choice(untried_actions))
        time = int(np.sum(self.counts)) + 1
        ucbs = self.c * np.sqrt(np.log(time) / self.counts)
        scores = self.estimates + ucbs
        best_actions = np.flatnonzero(scores == np.max(scores))
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: float) -> None:
        if action < 0 or action >= self.k:
            raise ValueError("action must be between 0 and k-1")

        self.counts[action] += 1

        if self.step_size is None:
            step = 1 / self.counts[action]
        else:
            step = self.step_size
        self.estimates[action] += step * (reward - self.estimates[action])
