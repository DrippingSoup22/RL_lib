"""Compare implemented bandit algorithms on Gymnasium bandit environments."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.bandits.environments import GaussianBandit
from rl_lib.algorithms.bandits import EpsilonGreedyBandits, UCBGreedyBandits


def run_condition(
    environment: str,
    algorithm: str,
    parameter: float,
    *,
    runs: int,
    steps: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rewards = np.empty((runs, steps))
    optimal = np.empty((runs, steps), dtype=bool)
    drift = 0.0 if environment == "stationary" else 0.01

    for run in range(runs):
        env = GaussianBandit(drift_std=drift)
        _, info = env.reset(seed=seed + run)
        if algorithm == "epsilon_greedy":
            agent = EpsilonGreedyBandits(
                k=env.k,
                epsilon=parameter,
                step_size=None if drift == 0 else 0.1,
                seed=seed + runs + run,
            )
        else:
            agent = UCBGreedyBandits(
                k=env.k,
                c=parameter,
                step_size=None if drift == 0 else 0.1,
                seed=seed + runs + run,
            )

        for step in range(steps):
            action = agent.select_action()
            _, reward, _, _, info = env.step(action)
            agent.update(action, reward)
            rewards[run, step] = reward
            optimal[run, step] = action == info["optimal_action"]
        env.close()

    return [
        {
            "environment": environment,
            "algorithm": algorithm,
            "parameter": parameter,
            "step": step + 1,
            "mean_reward": float(np.mean(rewards[:, step])),
            "optimal_action_rate": float(np.mean(optimal[:, step])),
        }
        for step in range(steps)
    ]


def run_experiment(runs: int, steps: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for environment in ("stationary", "nonstationary"):
        for algorithm, parameters in (
            ("epsilon_greedy", (0.01, 0.1)),
            ("ucb", (1.0, 2.0)),
        ):
            for parameter in parameters:
                rows.extend(
                    run_condition(
                        environment,
                        algorithm,
                        parameter,
                        runs=runs,
                        steps=steps,
                        seed=seed,
                    )
                )
    return rows


def write_outputs(
    output: Path,
    rows: list[dict[str, object]],
    *,
    runs: int,
    steps: int,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (output / "metadata.json").write_text(
        json.dumps({"runs": runs, "steps": steps, "seed": seed}, indent=2),
        encoding="utf-8",
    )

    late_start = max(0, steps - min(100, steps)) + 1
    lines = ["# Bandit experiment", "", "Late-training averages:", ""]
    groups = sorted({(r["environment"], r["algorithm"], r["parameter"]) for r in rows})
    for environment, algorithm, parameter in groups:
        selected = [
            r
            for r in rows
            if r["environment"] == environment
            and r["algorithm"] == algorithm
            and r["parameter"] == parameter
            and int(r["step"]) >= late_start
        ]
        reward = np.mean([float(r["mean_reward"]) for r in selected])
        optimal = np.mean([float(r["optimal_action_rate"]) for r in selected])
        lines.append(
            f"- {environment}, {algorithm}={parameter:g}: "
            f"reward={reward:.3f}, optimal-action rate={optimal:.3f}"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.runs < 1 or args.steps < 1:
        parser.error("runs and steps must be positive")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("runs/bandits") / run_id
    rows = run_experiment(args.runs, args.steps, args.seed)
    write_outputs(output, rows, runs=args.runs, steps=args.steps, seed=args.seed)
    print(output)


if __name__ == "__main__":
    main()
