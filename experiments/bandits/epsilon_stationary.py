"""Evaluate epsilon-greedy bandits over stationary parameter grids."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from experiments.bandits.evaluation import (
    BanditConditionResult,
    evaluate_bandit,
    mean_and_confidence_interval,
    write_convergence_csv,
    write_summary_csv,
)
from experiments.bandits.stationary_gaussian_bandit import StationaryGaussianBandit
from rl_lib.algorithms.tabular.epsilon_greedy_bandits import EpsilonGreedyBandits

MSE_PLOT_FLOOR = 1e-6


def run_grid(
    *,
    initial_values: list[float],
    epsilons: list[float],
    k: int,
    reward_std: float,
    steps: int,
    runs: int,
    seed: int,
) -> list[BanditConditionResult]:
    """Run every parameter pair using the same environment and agent seeds."""

    def make_environment(run_seed: int) -> StationaryGaussianBandit:
        return StationaryGaussianBandit(
            k=k,
            reward_std=reward_std,
            seed=run_seed,
        )

    results = []
    for initial_value in initial_values:
        for epsilon in epsilons:
            measurements = evaluate_bandit(
                environment_factory=make_environment,
                agent_factory=lambda run_seed, q=initial_value, e=epsilon: (
                    EpsilonGreedyBandits(
                        k=k,
                        epsilon=e,
                        initial_value=q,
                        seed=run_seed,
                    )
                ),
                runs=runs,
                steps=steps,
                seed=seed,
            )
            results.append(
                BanditConditionResult(
                    parameters={
                        "initial_value": initial_value,
                        "epsilon": epsilon,
                    },
                    measurements=measurements,
                )
            )
    return results


def _plot_with_confidence(
    axis: Axes,
    steps: np.ndarray,
    samples: np.ndarray,
    label: str,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> None:
    mean, lower, upper = mean_and_confidence_interval(samples)
    if lower_bound is not None:
        mean = np.maximum(mean, lower_bound)
        lower = np.maximum(lower, lower_bound)
    if upper_bound is not None:
        mean = np.minimum(mean, upper_bound)
        upper = np.minimum(upper, upper_bound)

    (line,) = axis.plot(steps, mean, label=label)
    axis.fill_between(steps, lower, upper, color=line.get_color(), alpha=0.14)


def write_grid_figure(
    path: Path,
    initial_values: list[float],
    epsilons: list[float],
    results: list[BanditConditionResult],
    reward_std: float,
) -> None:
    """Compare policy quality and estimation error across the parameter grid."""
    steps = np.arange(1, results[0].measurements.rewards.shape[1] + 1)
    figure, axes = plt.subplots(
        nrows=3,
        ncols=len(initial_values),
        figsize=(15, 11),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for value_index, initial_value in enumerate(initial_values):
        for epsilon_index, epsilon in enumerate(epsilons):
            result_index = value_index * len(epsilons) + epsilon_index
            measurements = results[result_index].measurements
            label = f"epsilon={epsilon:g}"
            _plot_with_confidence(
                axes[0, value_index],
                steps,
                measurements.rewards,
                label,
            )
            _plot_with_confidence(
                axes[1, value_index],
                steps,
                measurements.selected_optimal_action,
                label,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            _plot_with_confidence(
                axes[2, value_index],
                steps,
                measurements.estimation_mse,
                label,
                lower_bound=MSE_PLOT_FLOOR,
            )

        axes[0, value_index].set_title(f"Initial Q={initial_value:g}")
        axes[2, value_index].set_xlabel("Interaction step")

    axes[0, 0].set_ylabel("Mean reward")
    axes[1, 0].set_ylabel("Optimal-action rate")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[2, 0].set_ylabel("Mean estimation MSE")
    axes[2, 0].set_yscale("log")
    for axis in axes.flat:
        axis.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncols=len(epsilons),
    )
    figure.suptitle(
        "Epsilon-greedy exploration and initialization comparison "
        f"(stationary bandit, reward std={reward_std:g})",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
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
        default=[0.0, 0.01, 0.1],
        metavar=("E1", "E2", "E3"),
    )
    parser.add_argument(
        "--reward-std",
        type=float,
        default=1.0,
        help="Stationary reward noise standard deviation.",
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
    if any(not np.isfinite(initial_value) for initial_value in args.initial_values):
        raise ValueError("initial values must be finite")
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
    write_convergence_csv(run_dir / "convergence.csv", "epsilon_greedy", results)
    write_summary_csv(run_dir / "summary.csv", "epsilon_greedy", results)
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
