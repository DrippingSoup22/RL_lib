"""Observation normalization for neural reinforcement-learning agents."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ObservationNormalizer:
    """Normalize fixed-size observations using bounds or running moments."""

    MODES = ("none", "bounds", "running")

    def __init__(
        self,
        observation_size: int,
        mode: str = "none",
        *,
        low: ArrayLike | None = None,
        high: ArrayLike | None = None,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        if observation_size <= 0:
            raise ValueError("Observation size must be positive")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")

        self.observation_size = observation_size
        self.mode = mode
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.count = 0
        self.mean = np.zeros(observation_size, dtype=np.float64)
        self.squared_deviation_sum = np.zeros(observation_size, dtype=np.float64)

        self.low: NDArray[np.float32] | None = None
        self.high: NDArray[np.float32] | None = None
        if mode == "bounds":
            if low is None or high is None:
                raise ValueError("bounds normalization requires low and high")
            low_array = self._array(low, name="low")
            high_array = self._array(high, name="high")
            if not np.all(np.isfinite(low_array)) or not np.all(
                np.isfinite(high_array)
            ):
                raise ValueError("bounds normalization requires finite bounds")
            if np.any(high_array <= low_array):
                raise ValueError("every high bound must be greater than its low bound")
            self.low = low_array
            self.high = high_array
        elif low is not None or high is not None:
            raise ValueError("low and high are only used by bounds normalization")

    def _array(self, observation: ArrayLike, *, name: str) -> NDArray[np.float32]:
        result = np.asarray(observation, dtype=np.float32).reshape(-1)
        if result.shape != (self.observation_size,):
            raise ValueError(
                f"{name} must contain {self.observation_size} finite values"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(f"all {name} values must be finite")
        return result.copy()

    def normalize(
        self,
        observation: ArrayLike,
        *,
        update: bool = False,
    ) -> NDArray[np.float32]:
        """Return one owned normalized observation, optionally updating moments."""
        values = self._array(observation, name="observation")
        if update and self.mode == "running":
            # Welford's update accumulates stable running mean and variance.
            running_values = values.astype(np.float64)
            self.count += 1
            difference = running_values - self.mean
            self.mean += difference / self.count
            difference_after_update = running_values - self.mean
            self.squared_deviation_sum += difference * difference_after_update

        if self.mode == "none":
            return values
        if self.mode == "bounds":
            assert self.low is not None and self.high is not None
            center = (self.high + self.low) / 2.0
            half_range = (self.high - self.low) / 2.0
            return ((values - center) / half_range).astype(np.float32)

        if self.count == 0:
            return values
        variance = self.squared_deviation_sum / self.count
        normalized = (values.astype(np.float64) - self.mean) / np.sqrt(
            variance + self.epsilon
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        """Return an owned, serializable snapshot of the normalizer."""
        return {
            "observation_size": self.observation_size,
            "mode": self.mode,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "count": self.count,
            "mean": self.mean.tolist(),
            "squared_deviation_sum": self.squared_deviation_sum.tolist(),
            "low": None if self.low is None else self.low.tolist(),
            "high": None if self.high is None else self.high.tolist(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ObservationNormalizer:
        """Restore a normalizer from :meth:`state_dict` output."""
        normalizer = cls(
            int(state["observation_size"]),
            str(state["mode"]),
            low=state["low"],
            high=state["high"],
            epsilon=float(state["epsilon"]),
            clip=float(state["clip"]),
        )
        count = int(state["count"])
        if count < 0:
            raise ValueError("normalizer count must be nonnegative")
        mean = np.asarray(state["mean"], dtype=np.float64)
        squared_deviation_sum = np.asarray(
            state["squared_deviation_sum"], dtype=np.float64
        )
        expected_shape = (normalizer.observation_size,)
        if (
            mean.shape != expected_shape
            or squared_deviation_sum.shape != expected_shape
        ):
            raise ValueError("normalizer statistics have the wrong shape")
        if not np.all(np.isfinite(mean)) or not np.all(
            np.isfinite(squared_deviation_sum)
        ):
            raise ValueError("normalizer statistics must be finite")
        if np.any(squared_deviation_sum < 0):
            raise ValueError("normalizer squared deviations must be nonnegative")
        normalizer.count = count
        normalizer.mean = mean.copy()
        normalizer.squared_deviation_sum = squared_deviation_sum.copy()
        return normalizer
