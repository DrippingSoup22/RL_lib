"""Result records returned by policy-gradient algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOActionSample:
    """Action and frozen behavior-policy measurements collected for PPO."""

    action: int
    log_probability: float
    value: float


@dataclass(frozen=True)
class PPOUpdateResult:
    """Loss measurements from one PPO minibatch update."""

    actor_loss: float
    critic_loss: float
    entropy: float
