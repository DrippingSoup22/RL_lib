"""Loading and validation for repository environment configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENVIRONMENT_DIR = Path(__file__).with_name("environments")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Everything needed to construct and baseline one Gymnasium environment."""

    name: str
    gym_id: str
    family: str
    observation_space: str
    action_space: str
    description: str
    make_kwargs: dict[str, Any]
    baseline_seeds: tuple[int, ...]
    baseline_episodes: int


def load_environment(path: Path) -> EnvironmentConfig:
    """Load one environment definition and fail early on malformed input."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    baseline = raw["baseline"]
    config = EnvironmentConfig(
        name=raw["name"],
        gym_id=raw["gym_id"],
        family=raw["family"],
        observation_space=raw["observation_space"],
        action_space=raw["action_space"],
        description=raw["description"],
        make_kwargs=raw.get("make_kwargs", {}),
        baseline_seeds=tuple(baseline["seeds"]),
        baseline_episodes=baseline["episodes_per_seed"],
    )
    if not config.name or not config.gym_id:
        raise ValueError(f"name and gym_id are required in {path}")
    if not config.baseline_seeds or config.baseline_episodes < 1:
        raise ValueError(f"baseline settings are invalid in {path}")
    return config


def load_environments(name: str | None = None) -> list[EnvironmentConfig]:
    """Load all environment definitions, or the one with the requested name."""
    configs = [
        load_environment(path) for path in sorted(ENVIRONMENT_DIR.glob("*.json"))
    ]
    if name is None:
        return configs
    selected = [config for config in configs if config.name == name]
    if not selected:
        available = ", ".join(config.name for config in configs)
        raise ValueError(f"unknown environment {name!r}; choose from: {available}")
    return selected
