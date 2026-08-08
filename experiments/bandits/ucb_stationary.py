"""Evaluate UCB exploration constants on a stationary bandit."""

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
from rl_lib.algorithms.tabular.ucb_greedy_bandits import UCBGreedyBandits

MSE_PLOT_FLOOR = 1e-6


def run_sweep(
    *,
    exploration_constants: list[float],
    initial_value: float,
    k: int,
    reward_std: float,
    steps: int,
    runs: int,
    seed: int,
) -> list[BanditConditionResult]:
    """Run each exploration constant with paired environment and agent seeds."""

    def make_environment(run_seed: int) -> StationaryGaussianBandit:
        return StationaryGaussianBandit(
            k=k,
            reward_std=reward_std,
            seed=run_seed,
        )

    results = []
    for exploration_constant in exploration_constants:
        measurements = evaluate_bandit(
            environment_factory=make_environment,
            agent_factory=lambda run_seed, c=exploration_constant: UCBGreedyBandits(
                k=k,
                c=c,
                initial_value=initial_value,
                seed=run_seed,
            ),
            runs=runs,
            steps=steps,
            seed=seed,
        )
        results.append(
            BanditConditionResult(
                parameters={
                    "c": exploration_constant,
                    "initial_value": initial_value,
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


def write_figure(
    path: Path,
    exploration_constants: list[float],
    results: list[BanditConditionResult],
    reward_std: float,
    k: int,
) -> None:
    """Plot reward, optimal-action rate, and estimate error for each constant."""
    steps = np.arange(1, results[0].measurements.rewards.shape[1] + 1)
    figure, axes = plt.subplots(nrows=3, figsize=(10, 11), sharex=True)

    for result, exploration_constant in zip(
        results,
        exploration_constants,
        strict=True,
    ):
        measurements = result.measurements
        label = f"c={exploration_constant:g}"
        _plot_with_confidence(axes[0], steps, measurements.rewards, label)
        _plot_with_confidence(
            axes[1],
            steps,
            measurements.selected_optimal_action,
            label,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        _plot_with_confidence(
            axes[2],
            steps,
            measurements.estimation_mse,
            label,
            lower_bound=MSE_PLOT_FLOOR,
        )

    axes[0].set_ylabel("Mean reward")
    axes[1].set_ylabel("Optimal-action rate")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("Mean estimation MSE")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Interaction step")
    for axis in axes:
        axis.axvline(k, color="black", linestyle=":", alpha=0.45)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(
        f"UCB exploration comparison (stationary bandit, reward std={reward_std:g})"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="Number of arms.")
    parser.add_argument(
        "--exploration-constants",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0],
        metavar="C",
        help="Non-negative UCB exploration constants.",
    )
    parser.add_argument(
        "--initial-value",
        type=float,
        default=0.0,
        help=(
            "Initial estimate. Forced first visits make this irrelevant after each "
            "arm has been sampled."
        ),
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
        default=Path("runs/bandits/ucb_greedy"),
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
    if not np.isfinite(args.initial_value):
        raise ValueError("initial value must be finite")
    if len(set(args.exploration_constants)) != len(args.exploration_constants):
        raise ValueError("exploration constants must be unique")
    if any(
        not np.isfinite(exploration_constant) or exploration_constant < 0
        for exploration_constant in args.exploration_constants
    ):
        raise ValueError("exploration constants must be finite and non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results = run_sweep(
        exploration_constants=args.exploration_constants,
        initial_value=args.initial_value,
        k=args.k,
        reward_std=args.reward_std,
        steps=args.steps,
        runs=args.runs,
        seed=args.seed,
    )
    write_convergence_csv(run_dir / "convergence.csv", "ucb_greedy", results)
    write_summary_csv(run_dir / "summary.csv", "ucb_greedy", results)
    write_figure(
        run_dir / "ucb_c_sweep.png",
        args.exploration_constants,
        results,
        args.reward_std,
        args.k,
    )

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "experiment": "ucb_exploration_constant_sweep",
        "k": args.k,
        "initial_value": args.initial_value,
        "exploration_constants": args.exploration_constants,
        "reward_std": args.reward_std,
        "steps": args.steps,
        "runs": args.runs,
        "base_seed": args.seed,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote UCB exploration experiment to {run_dir}")


if __name__ == "__main__":
    main()
