"""Record one configured Gymnasium episode as an animated GIF."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
from PIL import Image

from experiments.config import EnvironmentConfig, load_environments

DEFAULT_ENVIRONMENT = "frozen_lake"


def default_output_path(environment: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("runs/previews") / f"{environment}-random-{timestamp}.gif"


def record_random_episode(
    config: EnvironmentConfig,
    output_path: Path,
    seed: int,
) -> float:
    """Record a random-policy episode and return its total reward."""
    # RGB-array recording must not depend on the host's WSLg or audio services.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = gym.make(config.gym_id, render_mode="rgb_array", **config.make_kwargs)
    frames: list[Image.Image] = []
    episode_return = 0.0
    try:
        env.action_space.seed(seed)
        env.reset(seed=seed)
        frames.append(Image.fromarray(env.render()))
        terminated = truncated = False

        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            episode_return += float(reward)
            frames.append(Image.fromarray(env.render()))
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 / env.metadata.get("render_fps", 30))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return episode_return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_environments(args.environment)[0]
    output_path = args.output or default_output_path(config.name)
    episode_return = record_random_episode(config, output_path, args.seed)
    print(f"Recorded return={episode_return:g} in {output_path.resolve()}")


if __name__ == "__main__":
    main()
