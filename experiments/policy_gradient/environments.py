"""Environment semantics for policy-gradient experiments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike

from rl_lib.data import ObservationNormalizer

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


class NormalizeObservation(gym.ObservationWrapper):
    """Apply one shared normalizer, updating it only for training observations."""

    def __init__(
        self,
        env: gym.Env,
        normalizer: ObservationNormalizer,
        *,
        update: bool,
    ) -> None:
        super().__init__(env)
        self.normalizer = normalizer
        self.update_statistics = update
        if normalizer.mode == "bounds":
            low = np.full(normalizer.observation_size, -1.0, dtype=np.float32)
            high = np.full(normalizer.observation_size, 1.0, dtype=np.float32)
        elif normalizer.mode == "running":
            low = np.full(
                normalizer.observation_size, -normalizer.clip, dtype=np.float32
            )
            high = np.full(
                normalizer.observation_size, normalizer.clip, dtype=np.float32
            )
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
            low = env.observation_space.low
            high = env.observation_space.high
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: object) -> np.ndarray:
        return self.normalizer.normalize(
            observation,
            update=self.update_statistics,
        )


class ScaleReward(gym.RewardWrapper):
    """Scale learning rewards by one positive constant."""

    def __init__(self, env: gym.Env, scale: float) -> None:
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return float(reward) * self.scale


def observation_normalizer(
    env: gym.Env,
    mode: str,
) -> ObservationNormalizer:
    """Build a normalizer matching one flattened Box observation space."""
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("observation normalization requires a Box observation space")
    observation_size = int(np.prod(env.observation_space.shape))
    if mode == "bounds":
        return ObservationNormalizer(
            observation_size,
            mode,
            low=env.observation_space.low,
            high=env.observation_space.high,
        )
    return ObservationNormalizer(observation_size, mode)


def prepare_environment(
    env: gym.Env,
    normalizer: ObservationNormalizer,
    *,
    update_normalization: bool,
    reward_scale: float = 1.0,
) -> gym.Env:
    """Attach shared observation preprocessing and optional reward scaling."""
    prepared = env
    if normalizer.mode != "none":
        prepared = NormalizeObservation(
            prepared,
            normalizer,
            update=update_normalization,
        )
    if reward_scale != 1.0:
        prepared = ScaleReward(prepared, reward_scale)
    return prepared


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
