"""Small training diagnostics for value-function approximation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class EpisodeDiagnostics:
    """Summarize the learning signals observed during one episode."""

    episode_return: float
    episode_length: int
    mean_absolute_td_error: float
    maximum_absolute_td_error: float
    mean_absolute_q_value: float
    maximum_absolute_q_value: float
    mean_action_gap: float
    dominant_action_fraction: float
    action_counts: tuple[int, ...]


class EpisodeDiagnosticTracker:
    """Accumulate compact diagnostics without retaining individual steps."""

    def __init__(self, number_of_actions: int) -> None:
        if number_of_actions <= 0:
            raise ValueError("number of actions must be positive")

        self.number_of_actions = number_of_actions
        self.episode_return = 0.0
        self.episode_length = 0
        self.absolute_td_error_sum = 0.0
        self.maximum_absolute_td_error = 0.0
        self.mean_absolute_q_value_sum = 0.0
        self.maximum_absolute_q_value = 0.0
        self.action_gap_sum = 0.0
        self.action_counts = [0] * number_of_actions

    def record_step(
        self,
        *,
        reward: float,
        action: int,
        td_error: float,
        action_values: ArrayLike,
    ) -> None:
        """Record one transition and its pre-update action values."""
        if not 0 <= action < self.number_of_actions:
            raise ValueError("action must stay inside the possible actions")

        values = np.asarray(action_values, dtype=float)
        if values.shape != (self.number_of_actions,):
            raise ValueError("action values must contain one value per action")

        absolute_td_error = abs(float(td_error))
        absolute_action_values = np.abs(values)
        if self.number_of_actions == 1:
            action_gap = 0.0
        else:
            ordered_values = np.sort(values)
            action_gap = float(ordered_values[-1] - ordered_values[-2])

        self.episode_return += float(reward)
        self.episode_length += 1
        self.absolute_td_error_sum += absolute_td_error
        self.maximum_absolute_td_error = max(
            self.maximum_absolute_td_error,
            absolute_td_error,
        )
        self.mean_absolute_q_value_sum += float(np.mean(absolute_action_values))
        self.maximum_absolute_q_value = max(
            self.maximum_absolute_q_value,
            float(np.max(absolute_action_values)),
        )
        self.action_gap_sum += action_gap
        self.action_counts[action] += 1

    def finish(self) -> EpisodeDiagnostics:
        """Return the episode summary after at least one recorded transition."""
        if self.episode_length == 0:
            raise ValueError("cannot finish diagnostics without a recorded step")

        return EpisodeDiagnostics(
            episode_return=self.episode_return,
            episode_length=self.episode_length,
            mean_absolute_td_error=(self.absolute_td_error_sum / self.episode_length),
            maximum_absolute_td_error=self.maximum_absolute_td_error,
            mean_absolute_q_value=(
                self.mean_absolute_q_value_sum / self.episode_length
            ),
            maximum_absolute_q_value=self.maximum_absolute_q_value,
            mean_action_gap=self.action_gap_sum / self.episode_length,
            dominant_action_fraction=(max(self.action_counts) / self.episode_length),
            action_counts=tuple(self.action_counts),
        )
