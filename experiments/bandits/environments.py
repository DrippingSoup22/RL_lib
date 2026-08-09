"""Gymnasium environments for stationary and drifting k-armed bandits."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class GaussianBandit(gym.Env[int, int]):
    """A one-state Gaussian bandit using the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        k: int = 10,
        reward_std: float = 1.0,
        drift_std: float = 0.0,
    ) -> None:
        if k < 1 or reward_std < 0 or drift_std < 0:
            raise ValueError("k must be positive and standard deviations non-negative")
        self.k = k
        self.reward_std = reward_std
        self.drift_std = drift_std
        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = gym.spaces.Discrete(1)
        self.action_values = np.zeros(k, dtype=float)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        if self.drift_std == 0:
            self.action_values = self.np_random.normal(size=self.k)
        else:
            self.action_values.fill(0.0)
        return 0, self._info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be between 0 and {self.k - 1}")
        reward = float(
            self.np_random.normal(self.action_values[action], self.reward_std)
        )
        info = self._info()
        if self.drift_std > 0:
            self.action_values += self.np_random.normal(
                scale=self.drift_std,
                size=self.k,
            )
        return 0, reward, False, False, info

    def _info(self) -> dict[str, Any]:
        return {
            "action_values": self.action_values.copy(),
            "optimal_action": int(np.argmax(self.action_values)),
        }
