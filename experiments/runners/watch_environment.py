"""Render a few episodes from one configured Gymnasium environment."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym

from experiments.config import EnvironmentConfig, load_environments

DEFAULT_ENVIRONMENT = "frozen_lake"


def resolve_render_mode(config: EnvironmentConfig, requested: str) -> str:
    """Prefer terminal rendering for toy-text tasks when mode is automatic."""
    if requested == "auto":
        return "ansi" if config.family == "toy_text" else "human"
    return requested


def show_terminal_frame(env: gym.Env) -> None:
    """Replace the previous ANSI frame and preserve the environment's FPS."""
    frame = env.render()
    if frame is not None:
        print(f"\033[2J\033[H{frame}", end="", flush=True)
        fps = env.metadata.get("render_fps", 4)
        time.sleep(1 / fps)


def watch_random_agent(
    config: EnvironmentConfig,
    episodes: int,
    seed: int,
    render_mode: str = "auto",
) -> list[float]:
    """Render random-policy episodes and return their total rewards."""
    returns: list[float] = []
    render_mode = resolve_render_mode(config, render_mode)
    env = gym.make(config.gym_id, render_mode=render_mode, **config.make_kwargs)
    try:
        env.action_space.seed(seed)
        for episode in range(episodes):
            _, _ = env.reset(seed=seed + episode)
            if render_mode == "ansi":
                show_terminal_frame(env)
            episode_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                _, reward, terminated, truncated, _ = env.step(
                    env.action_space.sample()
                )
                episode_return += float(reward)
                if render_mode == "ansi":
                    show_terminal_frame(env)

            returns.append(episode_return)
            print(f"Episode {episode + 1}: return={episode_return:g}")
    finally:
        env.close()
    return returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        default=DEFAULT_ENVIRONMENT,
        help=f"Configured environment name (default: {DEFAULT_ENVIRONMENT}).",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render-mode",
        choices=("auto", "human", "ansi", "rgb_array"),
        default="auto",
        help="Use terminal rendering for toy-text environments by default.",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    config = load_environments(args.environment)[0]
    watch_random_agent(
        config,
        episodes=args.episodes,
        seed=args.seed,
        render_mode=args.render_mode,
    )


if __name__ == "__main__":
    main()
