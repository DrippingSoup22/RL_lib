"""Evaluate epsilon-greedy bandits over initial-value and epsilon grids."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.bandits.stationary_gaussian_bandit import StationaryGaussianBandit
from rl_lib.algorithms.tabular.EpsilonGreedyBandits import EpsilonGreedyBandits

MSE_PLOT_FLOOR = 1e-6


@dataclass(frozen=True)
class GridResults:
    """Per-run measurements for each initial-value and epsilon pair."""

    estimation_mse: np.ndarray
    fraction_actions_tried: np.ndarray
    selected_optimal_action: np.ndarray
    rewards: np.ndarray


def run_grid(
    *,
    initial_values: list[float],
    epsilons: list[float],
    k: int,
    reward_std: float,
    steps: int,
    runs: int,
    seed: int,
) -> GridResults:
    """Run every parameter pair using the same environment seeds."""
    shape = (len(initial_values), len(epsilons), runs, steps)
    estimation_mse = np.empty(shape, dtype=float)
    fraction_actions_tried = np.empty(shape, dtype=float)
    selected_optimal_action = np.empty(shape, dtype=bool)
    rewards = np.empty(shape, dtype=float)

    for value_index, initial_value in enumerate(initial_values):
        for epsilon_index, epsilon in enumerate(epsilons):
            for run in range(runs):
                environment = StationaryGaussianBandit(
                    k=k,
                    reward_std=reward_std,
                    seed=seed + run,
                )
                agent = EpsilonGreedyBandits(
                    k=k,
                    epsilon=epsilon,
                    initial_value=initial_value,
                    seed=seed + runs + run,
                )

                for step_index in range(steps):
                    action = agent.select_action()
                    reward = environment.step(action)
                    agent.update(action, reward)

                    estimation_mse[value_index, epsilon_index, run, step_index] = (
                        np.mean((agent.estimates - environment.action_values) ** 2)
                    )
                    fraction_actions_tried[
                        value_index, epsilon_index, run, step_index
                    ] = np.count_nonzero(agent.counts) / k
                    selected_optimal_action[
                        value_index, epsilon_index, run, step_index
                    ] = action == environment.optimal_action
                    rewards[value_index, epsilon_index, run, step_index] = reward

    return GridResults(
        estimation_mse=estimation_mse,
        fraction_actions_tried=fraction_actions_tried,
        selected_optimal_action=selected_optimal_action,
        rewards=rewards,
    )


def write_convergence_csv(
    path: Path,
    initial_values: list[float],
    epsilons: list[float],
    results: GridResults,
) -> None:
    """Write aggregate measurements for every grid cell and step."""
    fields = (
        "initial_value",
        "epsilon",
        "step",
        "mean_estimation_mse",
        "std_estimation_mse",
        "mean_fraction_actions_tried",
        "optimal_action_rate",
        "mean_reward",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for value_index, initial_value in enumerate(initial_values):
            for epsilon_index, epsilon in enumerate(epsilons):
                for step_index in range(results.estimation_mse.shape[3]):
                    mse = results.estimation_mse[
                        value_index, epsilon_index, :, step_index
                    ]
                    writer.writerow(
                        {
                            "initial_value": initial_value,
                            "epsilon": epsilon,
                            "step": step_index + 1,
                            "mean_estimation_mse": float(np.mean(mse)),
                            "std_estimation_mse": float(np.std(mse)),
                            "mean_fraction_actions_tried": float(
                                np.mean(
                                    results.fraction_actions_tried[
                                        value_index, epsilon_index, :, step_index
                                    ]
                                )
                            ),
                            "optimal_action_rate": float(
                                np.mean(
                                    results.selected_optimal_action[
                                        value_index, epsilon_index, :, step_index
                                    ]
                                )
                            ),
                            "mean_reward": float(
                                np.mean(
                                    results.rewards[
                                        value_index, epsilon_index, :, step_index
                                    ]
                                )
                            ),
                        }
                    )


def write_summary_csv(
    path: Path,
    initial_values: list[float],
    epsilons: list[float],
    results: GridResults,
) -> None:
    """Write final aggregate measurements for all nine grid cells."""
    fields = (
        "initial_value",
        "epsilon",
        "final_mean_estimation_mse",
        "final_mean_fraction_actions_tried",
        "final_optimal_action_rate",
        "final_mean_reward",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for value_index, initial_value in enumerate(initial_values):
            for epsilon_index, epsilon in enumerate(epsilons):
                writer.writerow(
                    {
                        "initial_value": initial_value,
                        "epsilon": epsilon,
                        "final_mean_estimation_mse": float(
                            np.mean(
                                results.estimation_mse[
                                    value_index, epsilon_index, :, -1
                                ]
                            )
                        ),
                        "final_mean_fraction_actions_tried": float(
                            np.mean(
                                results.fraction_actions_tried[
                                    value_index, epsilon_index, :, -1
                                ]
                            )
                        ),
                        "final_optimal_action_rate": float(
                            np.mean(
                                results.selected_optimal_action[
                                    value_index, epsilon_index, :, -1
                                ]
                            )
                        ),
                        "final_mean_reward": float(
                            np.mean(results.rewards[value_index, epsilon_index, :, -1])
                        ),
                    }
                )


def write_grid_figure(
    path: Path,
    initial_values: list[float],
    epsilons: list[float],
    results: GridResults,
    reward_std: float,
) -> None:
    """Plot one estimation-error curve for each parameter pair."""
    steps = np.arange(1, results.estimation_mse.shape[3] + 1)
    figure, axes = plt.subplots(
        nrows=len(initial_values),
        ncols=len(epsilons),
        figsize=(13, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for value_index, initial_value in enumerate(initial_values):
        for epsilon_index, epsilon in enumerate(epsilons):
            axis = axes[value_index, epsilon_index]
            mean_mse = np.mean(
                results.estimation_mse[value_index, epsilon_index],
                axis=0,
            )
            axis.plot(steps, np.maximum(mean_mse, MSE_PLOT_FLOOR))
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            axis.set_title(f"epsilon={epsilon:g}")
            if epsilon_index == 0:
                axis.set_ylabel(f"initial Q={initial_value:g}\nMean MSE")
            if value_index == len(initial_values) - 1:
                axis.set_xlabel("Interaction step")

    figure.suptitle(
        "Epsilon-greedy estimate convergence "
        f"(stationary bandit, reward std={reward_std:g})"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="Number of arms.")
    parser.add_argument(
        "--initial-values",
        type=float,
        nargs=3,
        default=[0.0, 1.0, 5.0],
        metavar=("Q1", "Q2", "Q3"),
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs=3,
        default=[0.0, 0.1, 0.2],
        metavar=("E1", "E2", "E3"),
    )
    parser.add_argument(
        "--reward-std",
        type=float,
        default=0.0,
        help="Reward noise; zero gives deterministic rewards.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/bandits/epsilon_greedy"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.k < 1:
        raise ValueError("k must be at least 1")
    if args.reward_std < 0:
        raise ValueError("reward_std must be non-negative")
    if args.steps < 1:
        raise ValueError("steps must be at least 1")
    if args.runs < 1:
        raise ValueError("runs must be at least 1")
    if len(set(args.initial_values)) != 3:
        raise ValueError("the three initial values must be unique")
    if len(set(args.epsilons)) != 3:
        raise ValueError("the three epsilon values must be unique")
    if any(not 0.0 <= epsilon <= 1.0 for epsilon in args.epsilons):
        raise ValueError("epsilon values must be between 0 and 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results = run_grid(
        initial_values=args.initial_values,
        epsilons=args.epsilons,
        k=args.k,
        reward_std=args.reward_std,
        steps=args.steps,
        runs=args.runs,
        seed=args.seed,
    )
    write_convergence_csv(
        run_dir / "convergence.csv",
        args.initial_values,
        args.epsilons,
        results,
    )
    write_summary_csv(
        run_dir / "summary.csv",
        args.initial_values,
        args.epsilons,
        results,
    )
    write_grid_figure(
        run_dir / "epsilon_greedy_grid.png",
        args.initial_values,
        args.epsilons,
        results,
        args.reward_std,
    )

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "experiment": "epsilon_greedy_parameter_grid",
        "k": args.k,
        "initial_values": args.initial_values,
        "epsilons": args.epsilons,
        "reward_std": args.reward_std,
        "steps": args.steps,
        "runs": args.runs,
        "base_seed": args.seed,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote epsilon-greedy grid experiment to {run_dir}")


if __name__ == "__main__":
    main()
