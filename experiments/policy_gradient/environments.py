"""Environment semantics for policy-gradient experiments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike

KNOWN_ENVIRONMENTS = (
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Blackjack-v1",
    "CliffWalking-v1",
    "FrozenLake-v1",
    "Taxi-v4",
)


def make_environment(
    environment: str,
    *,
    render_mode: str | None = None,
    max_episode_steps: int | None = None,
) -> gym.Env:
    """Create an environment compatible with a supported neural policy."""
    kwargs = {"render_mode": render_mode}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    elif environment == "CliffWalking-v1":
        kwargs["max_episode_steps"] = 200
    env = gym.make(environment, **kwargs)
    if isinstance(env.action_space, gym.spaces.Discrete):
        if env.action_space.start != 0:
            env.close()
            raise ValueError("discrete environment actions must start at zero")
    elif isinstance(env.action_space, gym.spaces.Box):
        if len(env.action_space.shape) != 1:
            env.close()
            raise ValueError("continuous actions must be one-dimensional vectors")
        if not np.all(np.isfinite(env.action_space.low)) or not np.all(
            np.isfinite(env.action_space.high)
        ):
            env.close()
            raise ValueError("continuous action bounds must be finite")
    else:
        env.close()
        raise ValueError("environment must have a Discrete or Box action space")
    try:
        return gym.wrappers.FlattenObservation(env)
    except (NotImplementedError, ValueError) as error:
        env.close()
        raise ValueError(
            "environment observations must have a fixed-size flattenable space"
        ) from error


def experiment_defaults(environment: str, preset: str) -> dict[str, int]:
    if preset == "quick":
        return {
            "training_episodes": 5,
            "evaluation_episodes": 2,
            "seeds": 1,
        }
    if preset == "tuning":
        return {
            "training_episodes": 1_000 if environment != "CartPole-v1" else 500,
            "evaluation_episodes": 10,
            "seeds": 1,
        }
    return {
        "training_episodes": 2_000 if environment != "CartPole-v1" else 1_000,
        "evaluation_episodes": 50,
        "seeds": 3,
    }


def episode_succeeded(
    environment: str,
    *,
    terminated: bool,
    truncated: bool,
    final_reward: float,
) -> bool:
    if environment == "CartPole-v1":
        return truncated and not terminated
    if environment in ("Blackjack-v1", "FrozenLake-v1"):
        return terminated and final_reward > 0
    return terminated


def success_definition(environment: str) -> str:
    if environment == "CartPole-v1":
        return "reaches the time limit without true termination"
    if environment == "Acrobot-v1":
        return "true termination after the Acrobot reaches the target height"
    if environment == "MountainCar-v0":
        return "true termination after the car reaches the goal"
    if environment == "MountainCarContinuous-v0":
        return "true termination after the continuous-control car reaches the goal"
    if environment in ("Blackjack-v1", "FrozenLake-v1"):
        return "positive reward on true termination"
    if environment == "CliffWalking-v1":
        return "true termination after reaching the goal"
    if environment == "Taxi-v4":
        return "true termination after passenger delivery"
    return "true environment termination"


def action_name(environment: str, action: int | ArrayLike) -> str:
    if environment == "MountainCarContinuous-v0":
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        return "force [" + ", ".join(f"{value:.3g}" for value in values) + "]"

    labels = {
        "CartPole-v1": ("push left", "push right"),
        "Acrobot-v1": ("negative torque", "no torque", "positive torque"),
        "MountainCar-v0": ("push left", "no push", "push right"),
        "Blackjack-v1": ("stick", "hit"),
        "CliffWalking-v1": ("up", "right", "down", "left"),
        "FrozenLake-v1": ("left", "down", "right", "up"),
        "Taxi-v4": ("south", "north", "east", "west", "pickup", "drop-off"),
    }.get(environment)
    return str(action) if labels is None else labels[action]


def recording_frame_stride(environment: str) -> int:
    return 2 if environment.startswith("MountainCar") else 4
