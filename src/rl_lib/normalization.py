"""Observation normalisation for neural reinforcement-learning agents.

An observation is a tensor whose last dimension holds its values; any leading
dimensions form a batch, such as one observation per environment. The
normaliser keeps its bounds and statistics as tensors on one device.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


class ObservationNormalizer:
    """Rescale observations so that every value has a comparable size.

    ``none`` leaves observations unchanged. ``bounds`` maps each value's known
    range ``[low, high]`` to ``[-1, 1]``. ``running`` subtracts the mean and
    divides by the standard deviation of the observations seen so far, then
    clips to plus or minus ``clip_limit``; those statistics change only when
    ``normalize`` is asked to update them.

    The running statistics are ``observation_count``, the ``mean``, and
    ``squared_deviation_sum``, the sum of every observation's squared distance
    from the mean, so that the variance is ``squared_deviation_sum /
    observation_count``. They are float64 to stay precise over millions of
    observations.
    """

    MODES = ("none", "bounds", "running")

    def __init__(
        self,
        observation_size: int,
        mode: str = "none",
        *,
        low: torch.Tensor | None = None,
        high: torch.Tensor | None = None,
        epsilon: float = 1e-8,
        clip_limit: float = 10.0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Check the settings once and start with empty statistics.

        ``low`` and ``high`` are required in ``bounds`` mode and rejected in the
        others. ``epsilon`` keeps the division defined for a value that never
        varies.
        """
        if observation_size <= 0:
            raise ValueError("Observation size must be positive")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(clip_limit) or clip_limit <= 0:
            raise ValueError("clip_limit must be finite and positive")

        self.observation_size = observation_size
        self.mode = mode
        self.epsilon = float(epsilon)
        self.clip_limit = float(clip_limit)
        self.device = torch.device(device)

        # Running statistics, used only in ``running`` mode.
        self.observation_count = 0
        self.mean = torch.zeros(
            observation_size, dtype=torch.float64, device=self.device
        )
        self.squared_deviation_sum = torch.zeros_like(self.mean)

        # Fixed bounds, used only in ``bounds`` mode. They come from outside,
        # usually an environment's observation space, so they are checked once.
        self.low: torch.Tensor | None = None
        self.high: torch.Tensor | None = None
        if mode == "bounds":
            if low is None or high is None:
                raise ValueError("bounds normalization requires low and high")
            # Same device and dtype as the observations they will be used with.
            self.low = low.to(device=self.device, dtype=torch.float32)
            self.high = high.to(device=self.device, dtype=torch.float32)
            expected_shape = (observation_size,)
            if self.low.shape != expected_shape or self.high.shape != expected_shape:
                raise ValueError(
                    f"low and high must have {observation_size} values each"
                )
            if not (torch.isfinite(self.low).all() and torch.isfinite(self.high).all()):
                raise ValueError("low and high must be finite")
            if torch.any(self.high <= self.low):
                raise ValueError("every high bound must be greater than its low bound")
        elif low is not None or high is not None:
            raise ValueError("low and high are only used by bounds normalization")

    def normalize(
        self,
        observations: torch.Tensor,
        *,
        update_statistics: bool = False,
    ) -> torch.Tensor:
        """Normalised ``observations``, shape ``(..., observation_size)``.

        With ``update_statistics`` in ``running`` mode, the observations are
        first added to the running statistics, so they count towards their own
        normalisation. The result is a new tensor with the dtype of
        ``observations``, except in ``none`` mode, which returns the input.
        """
        if observations.shape[-1] != self.observation_size:
            raise ValueError(
                f"observations must have {self.observation_size} values each"
            )
        if update_statistics and self.mode == "running":
            self._add_to_statistics(observations)

        if self.mode == "none":
            return observations
        if self.mode == "bounds":
            assert self.low is not None and self.high is not None
            center = (self.high + self.low) / 2
            half_range = (self.high - self.low) / 2
            return (observations - center) / half_range

        # ``running``: nothing to normalise with until something was added.
        if self.observation_count == 0:
            return observations
        variance = self.squared_deviation_sum / self.observation_count
        standardized = (observations.double() - self.mean) / torch.sqrt(
            variance + self.epsilon
        )
        clipped = standardized.clamp(-self.clip_limit, self.clip_limit)
        return clipped.to(observations.dtype)

    def _add_to_statistics(self, observations: torch.Tensor) -> None:
        """Merge a batch of observations into the running statistics.

        The batch is summarised the same way as the statistics so far, by its
        count, mean, and sum of squared deviations from its own mean, and the
        two summaries are combined (the pairwise update of Chan, Golub, and
        LeVeque, 1979). Adding a batch at once gives the same statistics as
        adding its observations one by one.
        """
        batch = observations.reshape(-1, self.observation_size).double()
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        batch_squared_deviation_sum = (batch - batch_mean).square().sum(dim=0)

        total_count = self.observation_count + batch_count
        # How far the batch's mean lies from the mean so far; the new mean
        # moves towards it in proportion to the batch's share of the total.
        mean_shift = batch_mean - self.mean
        self.mean = self.mean + mean_shift * (batch_count / total_count)
        # The spread of each part around its own mean, plus the extra spread
        # that comes from the two means being apart.
        self.squared_deviation_sum = (
            self.squared_deviation_sum
            + batch_squared_deviation_sum
            + mean_shift.square() * (self.observation_count * batch_count / total_count)
        )
        self.observation_count = total_count

    def state_dict(self) -> dict[str, Any]:
        """The settings and statistics, as copies on the CPU, for checkpoints.

        Copies, so that a snapshot taken during training never changes while
        the statistics keep updating.
        """

        def cpu_copy(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return None if tensor is None else tensor.detach().to("cpu", copy=True)

        return {
            "observation_size": self.observation_size,
            "mode": self.mode,
            "epsilon": self.epsilon,
            "clip_limit": self.clip_limit,
            "observation_count": self.observation_count,
            "mean": cpu_copy(self.mean),
            "squared_deviation_sum": cpu_copy(self.squared_deviation_sum),
            "low": cpu_copy(self.low),
            "high": cpu_copy(self.high),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        device: torch.device | str = "cpu",
    ) -> ObservationNormalizer:
        """A new normalizer with the settings and statistics saved by
        ``state_dict``, on ``device``."""
        normalizer = cls(
            int(state["observation_size"]),
            str(state["mode"]),
            low=state["low"],
            high=state["high"],
            epsilon=float(state["epsilon"]),
            clip_limit=float(state["clip_limit"]),
            device=device,
        )
        normalizer.load_state_dict(state)
        return normalizer

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the running statistics saved by ``state_dict``.

        The settings stay this normalizer's own; the saved mode must match
        them, and the statistics must have this normalizer's size. The state
        comes from a file, so it is checked here, once.
        """
        if str(state["mode"]) != self.mode:
            raise ValueError("saved normalizer has a different mode")
        observation_count = int(state["observation_count"])
        mean, squared_deviation_sum = (
            torch.as_tensor(state[key], dtype=torch.float64, device=self.device)
            for key in ("mean", "squared_deviation_sum")
        )
        if observation_count < 0:
            raise ValueError("normalizer observation count must be nonnegative")
        expected_shape = (self.observation_size,)
        if (
            mean.shape != expected_shape
            or squared_deviation_sum.shape != expected_shape
        ):
            raise ValueError("normalizer statistics have the wrong shape")
        if not (
            torch.isfinite(mean).all() and torch.isfinite(squared_deviation_sum).all()
        ):
            raise ValueError("normalizer statistics must be finite")
        if torch.any(squared_deviation_sum < 0):
            raise ValueError("normalizer squared deviations must be nonnegative")

        # Clones, so the normaliser never shares memory with ``state``.
        self.observation_count = observation_count
        self.mean = mean.clone()
        self.squared_deviation_sum = squared_deviation_sum.clone()
