from __future__ import annotations

from typing import Any

import gymnasium as gym


class LoopingMDP(gym.Env[int, int]):
    """Small deterministic episodic MDP containing a reachable loop."""

    START = 0
    DECISION = 1
    SUCCESS = 2
    FAILURE = 3

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.Discrete(2)
        self.state: int | None = None
        self._terminated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset to the MDP's normal starting state."""
        super().reset(seed=seed)
        self.state = self.START
        self._terminated = False
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Apply one of the two deterministic actions."""
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        if self._terminated:
            raise RuntimeError("reset must be called after termination")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0 or 1")

        transitions = {
            (self.START, 0): (self.DECISION, 0.0),
            (self.START, 1): (self.FAILURE, -1.0),
            (self.DECISION, 0): (self.DECISION, -0.1),
            (self.DECISION, 1): (self.SUCCESS, 1.0),
        }
        next_state, reward = transitions[(self.state, action)]
        self.state = next_state
        self._terminated = next_state in (self.SUCCESS, self.FAILURE)

        return next_state, reward, self._terminated, False, {}
