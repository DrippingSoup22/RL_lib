"""Validated array representation of a continuous-observation rollout."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_lib.data.episode import EpisodeStep


@dataclass(frozen=True)
class RolloutArrays:
    """One bounded rollout converted to arrays for neural algorithms."""

    observations: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    final_state: NDArray[np.float32]


def rollout_arrays(
    steps: Sequence[EpisodeStep[NDArray[np.float32]]],
    final_state: ArrayLike,
    *,
    observation_size: int,
    number_of_actions: int | None = None,
) -> RolloutArrays:
    """Validate and convert rollout fields without applying RL semantics."""
    if not steps:
        raise ValueError("Rollout steps must not be empty; provide at least one step")
    if observation_size < 1:
        raise ValueError("Observation size must be positive")
    if number_of_actions is not None and number_of_actions < 1:
        raise ValueError("Number of actions must be positive")

    observations = np.asarray([step.state for step in steps], dtype=np.float32)
    if observations.shape != (len(steps), observation_size):
        raise ValueError("Rollout observations must match the model input size")
    if not np.all(np.isfinite(observations)):
        raise ValueError("Rollout observations must be finite")

    if any(
        isinstance(step.action, (bool, np.bool_))
        or not isinstance(step.action, (int, np.integer))
        for step in steps
    ):
        raise ValueError("Rollout actions must be integers")
    actions = np.asarray([step.action for step in steps], dtype=np.int64)
    if number_of_actions is not None and np.any(
        (actions < 0) | (actions >= number_of_actions)
    ):
        raise ValueError("Actions must stay inside the action space")

    rewards = np.asarray([step.reward for step in steps], dtype=np.float32)
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Rollout rewards must be finite")

    final_state_array = np.asarray(final_state, dtype=np.float32)
    if final_state_array.shape != (observation_size,):
        raise ValueError("final state must match the model input size")
    if not np.all(np.isfinite(final_state_array)):
        raise ValueError("final state must be finite")

    return RolloutArrays(observations, actions, rewards, final_state_array)
