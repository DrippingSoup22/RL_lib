"""Environment semantics for tabular temporal-difference experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from experiments.common import finite_state_encoding

KNOWN_ENVIRONMENTS = (
    "CliffWalking-v1",
    "FrozenLake-v1",
    "Blackjack-v1",
    "Taxi-v4",
)


def experiment_defaults(environment: str, preset: str) -> dict[str, int]:
    if preset == "quick":
        return {
            "prediction_episodes": 50,
            "training_episodes": 100,
            "evaluation_episodes": 10,
            "seeds": 1,
        }
    if preset == "tuning":
        return {
            "prediction_episodes": 250,
            "training_episodes": 500,
            "evaluation_episodes": 25,
            "seeds": 1,
        }
    if environment == "FrozenLake-v1":
        return {
            "prediction_episodes": 1_000,
            "training_episodes": 5_000,
            "evaluation_episodes": 200,
            "seeds": 3,
        }
    return {
        "prediction_episodes": 500,
        "training_episodes": 1_000,
        "evaluation_episodes": 100,
        "seeds": 3,
    }


def environment_configuration(
    environment: str,
    seed_values: tuple[int, ...],
    map_size: int,
    safe_probability: float,
    slippery: bool,
    max_episode_steps: int | None,
) -> tuple[list[dict[str, object]], int, int, int, Callable[[Any], int]]:
    if environment == "FrozenLake-v1":
        resolved_max_steps = max_episode_steps or (100 if map_size <= 4 else 200)
        maps = [
            generate_random_map(size=map_size, p=safe_probability, seed=seed)
            for seed in seed_values
        ]
        configurations = [
            {
                "desc": map_description,
                "is_slippery": slippery,
                "max_episode_steps": resolved_max_steps,
            }
            for map_description in maps
        ]
        states, encoder = finite_state_encoding(gym.spaces.Discrete(map_size**2))
        return configurations, states, 4, resolved_max_steps, encoder

    inspection = gym.make(environment)
    try:
        observation_space = inspection.observation_space
        action_space = inspection.action_space
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("environment must have a discrete action space")
        if action_space.start != 0:
            raise ValueError("environment must use zero-based discrete actions")
        default_max_steps = (
            inspection.spec.max_episode_steps
            if inspection.spec is not None
            and inspection.spec.max_episode_steps is not None
            else 200
        )
        states, encoder = finite_state_encoding(observation_space)
        actions = int(action_space.n)
    finally:
        inspection.close()

    resolved_max_steps = max_episode_steps or int(default_max_steps)
    configurations = [
        {"max_episode_steps": resolved_max_steps} for _seed in seed_values
    ]
    return configurations, states, actions, resolved_max_steps, encoder


def make_environment(
    environment: str,
    configuration: dict[str, object],
    *,
    render_mode: str | None = None,
) -> gym.Env:
    kwargs = dict(configuration)
    max_episode_steps = int(kwargs.pop("max_episode_steps"))
    return gym.make(
        environment,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


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


def action_name(environment: str, action: int) -> str:
    labels = {
        "Blackjack-v1": ("stick", "hit"),
        "CliffWalking-v1": ("up", "right", "down", "left"),
        "FrozenLake-v1": ("left", "down", "right", "up"),
        "Taxi-v4": ("south", "north", "east", "west", "pickup", "drop-off"),
    }.get(environment)
    return str(action) if labels is None else labels[action]


def episode_succeeded(environment: str, terminated: bool, reward: float) -> bool:
    if environment in ("CliffWalking-v1", "Taxi-v4"):
        return terminated
    return terminated and reward > 0


def success_definition(environment: str) -> str:
    if environment == "CliffWalking-v1":
        return "true termination after reaching the goal"
    if environment == "Taxi-v4":
        return "true termination after passenger delivery"
    return "positive reward on true termination"
