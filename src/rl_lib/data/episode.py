"""Episode data structures for episodic reinforcement-learning algorithms."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class EpisodeStep(Generic[StateT]):
    """One state, environment action, and resulting reward from an episode."""

    state: StateT
    action: int | NDArray[np.float32]
    reward: float
    # Continuous policies retain their unsquashed action for stable reevaluation.
    # Categorical rollouts may omit this because both action forms are identical.
    policy_action: int | NDArray[np.float32] | None = None


@dataclass(frozen=True)
class Episode(Generic[StateT]):
    """A completed sequence of interactions and its stopping condition."""

    steps: tuple[EpisodeStep[StateT], ...]
    final_state: StateT
    terminated: bool
    truncated: bool


def discounted_returns(
    rewards: ArrayLike,
    discount: float,
    *,
    bootstrap: float = 0.0,
) -> NDArray[np.float64]:
    """Calculate one return per reward, optionally bootstrapping after the last."""
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be finite and between 0 and 1")
    if not np.isfinite(bootstrap):
        raise ValueError("bootstrap must be finite")

    reward_values = np.asarray(rewards, dtype=float)
    if reward_values.ndim != 1:
        raise ValueError("rewards must be one-dimensional")

    returns = np.zeros(reward_values.size, dtype=float)
    running_return = float(bootstrap)
    for index in range(reward_values.size - 1, -1, -1):
        running_return = reward_values[index] + discount * running_return
        returns[index] = running_return
    return returns
