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
        self.td_error_count = 0
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
        self.record_transition(
            reward=reward,
            action=action,
            action_values=action_values,
        )
        self.record_td_errors((td_error,))

    def record_transition(
        self,
        *,
        reward: float,
        action: int,
        action_values: ArrayLike,
    ) -> None:
        """Record experience independently of its delayed n-step update."""
        if not 0 <= action < self.number_of_actions:
            raise ValueError("action must stay inside the possible actions")

        values = np.asarray(action_values, dtype=float)
        if values.shape != (self.number_of_actions,):
            raise ValueError("action values must contain one value per action")

        absolute_action_values = np.abs(values)
        if self.number_of_actions == 1:
            action_gap = 0.0
        else:
            ordered_values = np.sort(values)
            action_gap = float(ordered_values[-1] - ordered_values[-2])

        self.episode_return += float(reward)
        self.episode_length += 1
        self.mean_absolute_q_value_sum += float(np.mean(absolute_action_values))
        self.maximum_absolute_q_value = max(
            self.maximum_absolute_q_value,
            float(np.max(absolute_action_values)),
        )
        self.action_gap_sum += action_gap
        self.action_counts[action] += 1

    def record_td_errors(self, td_errors: ArrayLike) -> None:
        """Record one or more errors emitted by delayed n-step updates."""
        errors = np.asarray(td_errors, dtype=float)
        if errors.ndim != 1 or not np.all(np.isfinite(errors)):
            raise ValueError("TD errors must be a finite one-dimensional sequence")
        absolute_errors = np.abs(errors)
        self.absolute_td_error_sum += float(np.sum(absolute_errors))
        if absolute_errors.size:
            self.maximum_absolute_td_error = max(
                self.maximum_absolute_td_error,
                float(np.max(absolute_errors)),
            )
        self.td_error_count += int(errors.size)

    def finish(self) -> EpisodeDiagnostics:
        """Return the episode summary after at least one recorded transition."""
        if self.episode_length == 0:
            raise ValueError("cannot finish diagnostics without a recorded step")
        if self.td_error_count != self.episode_length:
            raise ValueError("each recorded transition must have one TD update")

        return EpisodeDiagnostics(
            episode_return=self.episode_return,
            episode_length=self.episode_length,
            mean_absolute_td_error=(self.absolute_td_error_sum / self.td_error_count),
            maximum_absolute_td_error=self.maximum_absolute_td_error,
            mean_absolute_q_value=(
                self.mean_absolute_q_value_sum / self.episode_length
            ),
            maximum_absolute_q_value=self.maximum_absolute_q_value,
            mean_action_gap=self.action_gap_sum / self.episode_length,
            dominant_action_fraction=(max(self.action_counts) / self.episode_length),
            action_counts=tuple(self.action_counts),
        )
