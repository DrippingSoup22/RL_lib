"""Validate configured environments and record random-policy episode metrics."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym

from experiments.config import EnvironmentConfig, load_environments

EPISODE_FIELDS = (
    "run_id",
    "algorithm",
    "environment",
    "gym_id",
    "seed",
    "phase",
    "episode",
    "return",
    "length",
    "terminated",
    "truncated",
)


def run_environment(config: EnvironmentConfig, run_id: str) -> list[dict[str, Any]]:
    """Collect random-policy episodes for one configured environment."""
    records: list[dict[str, Any]] = []
    env = gym.make(config.gym_id, **config.make_kwargs)
    try:
        for seed in config.baseline_seeds:
            env.action_space.seed(seed)
            for episode in range(config.baseline_episodes):
                _, _ = env.reset(seed=seed + episode)
                episode_return = 0.0
                episode_length = 0
                terminated = truncated = False

                while not (terminated or truncated):
                    action = env.action_space.sample()
                    _, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)
                    episode_length += 1

                records.append(
                    {
                        "run_id": run_id,
                        "algorithm": "random",
                        "environment": config.name,
                        "gym_id": config.gym_id,
                        "seed": seed,
                        "phase": "evaluation",
                        "episode": episode,
                        "return": episode_return,
                        "length": episode_length,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
    finally:
        env.close()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        help="Run one configured environment by name; the default runs all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/random_baseline"),
        help="Parent directory for the timestamped raw run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_environments(args.environment)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    records = [
        record
        for config in configs
        for record in run_environment(config=config, run_id=run_id)
    ]
    with (run_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "run_id": run_id,
        "algorithm": "random",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "environments": [config.name for config in configs],
        "episode_count": len(records),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {len(records)} episodes in {run_dir}")


if __name__ == "__main__":
    main()
