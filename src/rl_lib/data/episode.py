"""Episode data structures for episodic reinforcement-learning algorithms."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class EpisodeStep:
    """One state, action, and resulting reward from an episode."""

    state: int
    action: int
    reward: float


@dataclass(frozen=True)
class Episode:
    """A completed sequence of interactions and its stopping condition."""

    steps: tuple[EpisodeStep, ...]
    final_state: int
    terminated: bool
    truncated: bool


def discounted_returns(
    rewards: ArrayLike,
    discount: float,
) -> NDArray[np.float64]:
    """Calculate one return for every reward position in an episode."""
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be finite and between 0 and 1")

    reward_values = np.asarray(rewards, dtype=float)
    if reward_values.ndim != 1:
        raise ValueError("rewards must be one-dimensional")

    returns = np.zeros(reward_values.size, dtype=float)
    running_return = 0.0
    for index in range(reward_values.size - 1, -1, -1):
        running_return = reward_values[index] + discount * running_return
        returns[index] = running_return
    return returns
