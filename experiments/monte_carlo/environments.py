"""Environment semantics for tabular Monte Carlo experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from experiments.common import finite_state_encoding

BLACKJACK_STATES = 32 * 11 * 2
BLACKJACK_ACTIONS = 2
KNOWN_ENVIRONMENTS = (
    "Blackjack-v1",
    "CliffWalking-v1",
    "FrozenLake-v1",
    "Taxi-v4",
)
RECORDING_SEEDS = (4, 5, 6, 7, 8)


def experiment_defaults(environment: str, preset: str) -> dict[str, int]:
    if preset == "quick":
        return {
            "prediction_episodes": 100 if environment != "Blackjack-v1" else 500,
            "training_episodes": 250 if environment != "Blackjack-v1" else 500,
            "evaluation_episodes": 25 if environment != "Blackjack-v1" else 100,
            "seeds": 1,
        }
    if preset == "tuning":
        return {
            "prediction_episodes": 2_000,
            "training_episodes": 5_000,
            "evaluation_episodes": 100,
            "seeds": 1,
        }
    if environment == "Blackjack-v1":
        return {
            "prediction_episodes": 50_000,
            "training_episodes": 50_000,
            "evaluation_episodes": 5_000,
            "seeds": 3,
        }
    if environment == "Taxi-v4":
        return {
            "prediction_episodes": 2_000,
            "training_episodes": 10_000,
            "evaluation_episodes": 100,
            "seeds": 3,
        }
    return {
        "prediction_episodes": 2_000,
        "training_episodes": 5_000,
        "evaluation_episodes": 250,
        "seeds": 3,
    }


def recording_settings(environment: str) -> tuple[tuple[int, ...], int, int]:
    if environment == "Blackjack-v1":
        return RECORDING_SEEDS, 750, 1_500
    return RECORDING_SEEDS[:1], 150, 1_000


def action_name(environment: str, action: int) -> str:
    labels = {
        "Blackjack-v1": ("stick", "hit"),
        "CliffWalking-v1": ("up", "right", "down", "left"),
        "FrozenLake-v1": ("left", "down", "right", "up"),
        "Taxi-v4": ("south", "north", "east", "west", "pickup", "drop-off"),
    }.get(environment)
    return str(action) if labels is None else labels[action]


def encode_blackjack_state(observation: tuple[int, int, int]) -> int:
    player, dealer, usable_ace = observation
    return (player * 11 + dealer) * 2 + int(usable_ace)


def encode_discrete_state(observation: Any) -> int:
    return int(observation)


def make_environment(
    environment: str,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    kwargs: dict[str, object] = {"render_mode": render_mode}
    if environment == "CliffWalking-v1":
        kwargs["max_episode_steps"] = 200
    return gym.make(environment, **kwargs)


def inspect_environment(
    environment: str,
) -> tuple[int, int, Callable[[Any], int], bool]:
    env = make_environment(environment)
    try:
        renderable = "rgb_array" in env.metadata.get("render_modes", [])
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(f"{environment} must have a Discrete action space")
        if action_space.start != 0:
            raise ValueError(f"{environment} must use zero-based discrete actions")
        number_of_states, encoder = finite_state_encoding(env.observation_space)
        return (
            number_of_states,
            int(action_space.n),
            encoder,
            renderable,
        )
    finally:
        env.close()


def fixed_prediction_action(
    environment: str,
    rng: np.random.Generator,
    state: int,
    number_of_actions: int,
) -> int:
    if environment == "Blackjack-v1":
        player = state // 22
        return 0 if player >= 20 else 1
    if environment == "CliffWalking-v1":
        row, column = divmod(state, 12)
        if row == 3 and column < 11:
            return 0
        if column < 11:
            return 1
        return 2
    return int(rng.integers(number_of_actions))


def episode_succeeded(
    environment: str,
    *,
    terminated: bool,
    final_reward: float,
) -> bool | None:
    if environment == "CliffWalking-v1":
        return terminated
    if environment in ("Blackjack-v1", "FrozenLake-v1", "Taxi-v4"):
        return terminated and final_reward > 0
    return None


def success_definition(environment: str) -> str | None:
    if environment == "CliffWalking-v1":
        return "true termination after reaching the goal"
    if environment in ("Blackjack-v1", "FrozenLake-v1", "Taxi-v4"):
        return "positive reward on true termination"
    return None
