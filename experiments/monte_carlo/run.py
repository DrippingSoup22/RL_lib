"""Run Monte Carlo prediction and control on Gymnasium Blackjack."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_lib.algorithms.monte_carlo import (
    EveryVisitMonteCarloControl,
    EveryVisitMonteCarloPrediction,
    FirstVisitMonteCarloControl,
    FirstVisitMonteCarloPrediction,
)
from rl_lib.data.episode import Episode, EpisodeStep

NUMBER_OF_STATES = 32 * 11 * 2
NUMBER_OF_ACTIONS = 2


def encode_state(observation: tuple[int, int, int]) -> int:
    player, dealer, usable_ace = observation
    return (player * 11 + dealer) * 2 + int(usable_ace)


def generate_episode(
    env: gym.Env,
    select_action: Callable[[int], int],
    *,
    seed: int,
) -> Episode:
    observation, _ = env.reset(seed=seed)
    state = encode_state(observation)
    steps: list[EpisodeStep] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = select_action(state)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        steps.append(EpisodeStep(state, action, float(reward)))
        state = encode_state(next_observation)
    return Episode(tuple(steps), state, terminated, truncated)


def fixed_blackjack_action(state: int) -> int:
    player = state // 22
    return 0 if player >= 20 else 1


def greedy_action(
    action_values: np.ndarray,
    rng: np.random.Generator,
    state: int,
) -> int:
    values = action_values[state]
    return int(rng.choice(np.flatnonzero(values == np.max(values))))


def run_prediction(episodes: int, seed: int) -> list[dict[str, object]]:
    first = FirstVisitMonteCarloPrediction(NUMBER_OF_STATES)
    every = EveryVisitMonteCarloPrediction(NUMBER_OF_STATES)
    env = gym.make("Blackjack-v1")
    for episode in range(episodes):
        trajectory = generate_episode(
            env,
            fixed_blackjack_action,
            seed=seed + episode,
        )
        first.update(trajectory)
        every.update(trajectory)
    env.close()

    reference_states = ((20, 10, 0), (20, 10, 1), (13, 2, 0), (13, 2, 1))
    rows = []
    for observation in reference_states:
        state = encode_state(observation)
        for name, estimator in (("first_visit", first), ("every_visit", every)):
            rows.append(
                {
                    "phase": "prediction",
                    "algorithm": name,
                    "state": str(observation),
                    "value": float(estimator.values[state]),
                    "visits": int(estimator.visit_counts[state]),
                    "mean_return": "",
                    "win_rate": "",
                }
            )
    return rows


def run_control(
    training_episodes: int,
    evaluation_episodes: int,
    epsilon: float,
    seeds: int,
) -> list[dict[str, object]]:
    rows = []
    classes = (
        ("first_visit", FirstVisitMonteCarloControl),
        ("every_visit", EveryVisitMonteCarloControl),
    )
    for name, control_class in classes:
        returns = []
        wins = []
        for seed in range(seeds):
            env = gym.make("Blackjack-v1")
            agent = control_class(
                NUMBER_OF_STATES,
                NUMBER_OF_ACTIONS,
                epsilon=epsilon,
                seed=seed,
            )
            for episode in range(training_episodes):
                trajectory = generate_episode(
                    env,
                    agent.select_action,
                    seed=seed * training_episodes + episode,
                )
                agent.update(trajectory)

            evaluation_policy = partial(
                greedy_action,
                agent.action_values,
                np.random.default_rng(seed),
            )

            for episode in range(evaluation_episodes):
                trajectory = generate_episode(
                    env,
                    evaluation_policy,
                    seed=1_000_000 + seed * evaluation_episodes + episode,
                )
                episode_return = sum(step.reward for step in trajectory.steps)
                returns.append(episode_return)
                wins.append(episode_return > 0)
            env.close()
        rows.append(
            {
                "phase": "control",
                "algorithm": name,
                "state": "",
                "value": "",
                "visits": "",
                "mean_return": float(np.mean(returns)),
                "win_rate": float(np.mean(wins)),
            }
        )
    return rows


def write_outputs(output: Path, rows: list[dict[str, object]], metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    lines = ["# Monte Carlo on Blackjack-v1", ""]
    for row in rows:
        if row["phase"] == "prediction":
            lines.append(
                f"- prediction {row['algorithm']} {row['state']}: "
                f"V={float(row['value']):.3f}, visits={row['visits']}"
            )
        else:
            lines.append(
                f"- control {row['algorithm']}: "
                f"return={float(row['mean_return']):.3f}, "
                f"win rate={float(row['win_rate']):.3f}"
            )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-episodes", type=int, default=50_000)
    parser.add_argument("--training-episodes", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=5_000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    if (
        min(
            args.prediction_episodes,
            args.training_episodes,
            args.evaluation_episodes,
            args.seeds,
        )
        < 1
    ):
        parser.error("episode counts and seeds must be positive")

    rows = run_prediction(args.prediction_episodes, seed=0)
    rows.extend(
        run_control(
            args.training_episodes,
            args.evaluation_episodes,
            args.epsilon,
            args.seeds,
        )
    )
    metadata = vars(args) | {"environment": "Blackjack-v1"}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("runs/monte_carlo") / run_id
    write_outputs(output, rows, metadata)
    print(output)


if __name__ == "__main__":
    main()
