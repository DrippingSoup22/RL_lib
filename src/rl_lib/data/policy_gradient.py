"""Result records returned by policy-gradient algorithms."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ContinuousPPOActionSample:
    """Bounded action and frozen measurements collected by continuous PPO."""

    action: NDArray[np.float32]
    latent_action: NDArray[np.float32]
    log_probability: float
    value: float


@dataclass(frozen=True)
class CategoricalPPOActionSample:
    """Categorical action and frozen behavior measurements collected by PPO."""

    action: int
    log_probability: float
    value: float


@dataclass(frozen=True)
class PPOUpdateResult:
    """Loss measurements from one PPO minibatch update."""

    actor_loss: float
    critic_loss: float
    entropy: float
