"""Environment semantics for function-approximation experiments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

KNOWN_ENVIRONMENTS = (
    "Acrobot-v1",
    "MountainCar-v0",
    "CartPole-v1",
    "Blackjack-v1",
    "CliffWalking-v1",
    "FrozenLake-v1",
    "Taxi-v4",
)
TOY_TEXT_ENVIRONMENTS = frozenset(
    ("Blackjack-v1", "CliffWalking-v1", "FrozenLake-v1", "Taxi-v4")
)


def make_environment(
    environment: str,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a flat Box-observation, discrete-action Gymnasium environment."""
    kwargs: dict[str, object] = {"render_mode": render_mode}
    if environment == "CliffWalking-v1":
        kwargs["max_episode_steps"] = 200
    env = gym.make(environment, **kwargs)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise ValueError("environment must have a discrete action space")
    if env.action_space.start != 0:
        env.close()
        raise ValueError("environment actions must start at zero")

    original_space = env.observation_space
    try:
        if (
            isinstance(original_space, gym.spaces.Box)
            and np.all(np.isfinite(original_space.low))
            and np.all(np.isfinite(original_space.high))
        ):
            env = gym.wrappers.RescaleObservation(
                env,
                np.float32(-1.0),
                np.float32(1.0),
            )
        env = gym.wrappers.FlattenObservation(env)
    except (NotImplementedError, ValueError) as error:
        env.close()
        raise ValueError(
            "environment observations must have a fixed-size flattenable space"
        ) from error
    return env


def experiment_defaults(environment: str, preset: str) -> dict[str, int]:
    if preset == "quick":
        return {
            "training_episodes": 10 if environment == "MountainCar-v0" else 5,
            "evaluation_episodes": 2,
            "seeds": 1,
        }
    if preset == "tuning":
        return {
            "training_episodes": 500 if environment == "MountainCar-v0" else 250,
            "evaluation_episodes": 10,
            "seeds": 1,
        }
    return {
        "training_episodes": 2_000 if environment == "MountainCar-v0" else 1_000,
        "evaluation_episodes": 50,
        "seeds": 3,
    }


def progress_value(
    environment: str,
    observation: np.ndarray,
    episode_length: int,
) -> float:
    if environment == "MountainCar-v0":
        scaled_position = float(observation[0])
        return -1.2 + 0.9 * (scaled_position + 1.0)
    if environment == "Acrobot-v1":
        cos_first, sin_first, cos_second, sin_second = map(float, observation[:4])
        cos_combined = cos_first * cos_second - sin_first * sin_second
        return -cos_first - cos_combined
    return float(episode_length)


def progress_configuration(
    environment: str,
    max_episode_steps: int | None = None,
) -> tuple[str, float, str]:
    if environment == "MountainCar-v0":
        return (
            "Maximum position",
            0.5,
            "true termination after reaching position 0.5",
        )
    if environment == "Acrobot-v1":
        return (
            "Maximum tip height",
            1.0,
            "true termination after tip height exceeds 1.0",
        )
    if environment == "CartPole-v1":
        resolved_max_steps = 500 if max_episode_steps is None else max_episode_steps
        return (
            "Episode length",
            float(resolved_max_steps),
            "reaches the time limit without true termination",
        )
    if environment in ("Blackjack-v1", "FrozenLake-v1"):
        return (
            "Episode return",
            1.0,
            "positive reward on true termination",
        )
    if environment == "CliffWalking-v1":
        return (
            "Episode return",
            0.0,
            "true termination after reaching the goal",
        )
    if environment == "Taxi-v4":
        return (
            "Episode return",
            0.0,
            "true termination after passenger delivery",
        )
    resolved_max_steps = 1 if max_episode_steps is None else max_episode_steps
    return (
        "Episode length",
        float(resolved_max_steps),
        "reaches the configured time limit",
    )


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


def action_name(environment: str, action: int) -> str:
    labels = {
        "MountainCar-v0": ("push left", "no push", "push right"),
        "Acrobot-v1": ("negative torque", "no torque", "positive torque"),
        "CartPole-v1": ("push left", "push right"),
        "Blackjack-v1": ("stick", "hit"),
        "CliffWalking-v1": ("up", "right", "down", "left"),
        "FrozenLake-v1": ("left", "down", "right", "up"),
        "Taxi-v4": ("south", "north", "east", "west", "pickup", "drop-off"),
    }.get(environment)
    return str(action) if labels is None else labels[action]


def recording_frame_stride(environment: str) -> int:
    return 2 if environment == "MountainCar-v0" else 4


def episode_progress(
    environment: str,
    maximum_progress: float,
    episode_return: float,
) -> float:
    """Use return where Toy Text observations have no physical progress measure."""
    return episode_return if environment in TOY_TEXT_ENVIRONMENTS else maximum_progress
