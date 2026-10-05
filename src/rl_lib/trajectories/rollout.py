"""Validated array representation of a continuous-observation rollout."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_lib.trajectories.episode import EpisodeStep


@dataclass(frozen=True)
class RolloutArrays:
    """One bounded rollout converted to arrays for neural algorithms."""

    observations: NDArray[np.float32]
    actions: NDArray[np.int64] | NDArray[np.float32]
    rewards: NDArray[np.float32]
    final_state: NDArray[np.float32]


def _rollout_fields(
    steps: Sequence[EpisodeStep[NDArray[np.float32]]],
    final_state: ArrayLike,
    observation_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Validate fields shared by every continuous-observation rollout."""
    if not steps:
        raise ValueError("Rollout steps must not be empty; provide at least one step")
    if observation_size < 1:
        raise ValueError("Observation size must be positive")

    observations = np.asarray([step.state for step in steps], dtype=np.float32)
    if observations.shape != (len(steps), observation_size):
        raise ValueError("Rollout observations must match the model input size")
    if not np.all(np.isfinite(observations)):
        raise ValueError("Rollout observations must be finite")

    rewards = np.asarray([step.reward for step in steps], dtype=np.float32)
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Rollout rewards must be finite")

    final_state_array = np.asarray(final_state, dtype=np.float32)
    if final_state_array.shape != (observation_size,):
        raise ValueError("final state must match the model input size")
    if not np.all(np.isfinite(final_state_array)):
        raise ValueError("final state must be finite")

    return observations, rewards, final_state_array


def rollout_arrays(
    steps: Sequence[EpisodeStep[NDArray[np.float32]]],
    final_state: ArrayLike,
    *,
    observation_size: int,
    number_of_actions: int | None = None,
    action_size: int | None = None,
) -> RolloutArrays:
    """Validate and convert a discrete or continuous-action rollout."""
    if number_of_actions is not None and action_size is not None:
        raise ValueError("Provide either a discrete or continuous action size")

    observations, rewards, final_state_array = _rollout_fields(
        steps,
        final_state,
        observation_size,
    )
    if action_size is None:
        if number_of_actions is not None and number_of_actions < 1:
            raise ValueError("Number of actions must be positive")
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
    else:
        if action_size < 1:
            raise ValueError("Action size must be positive")
        # The bounded environment action cannot reliably recover its latent
        # Gaussian sample near tanh saturation, so the latent value is required.
        if any(step.policy_action is None for step in steps):
            raise ValueError("Continuous rollouts must store latent policy actions")
        try:
            actions = np.asarray(
                [step.policy_action for step in steps],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Latent policy actions must be numeric") from error
        expected_shape = (len(steps), action_size)
        if actions.shape != expected_shape:
            raise ValueError(f"Latent policy actions must have shape {expected_shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Latent policy actions must be finite")

    return RolloutArrays(observations, actions, rewards, final_state_array)
