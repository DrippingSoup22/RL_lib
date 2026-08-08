"""Compare standard UCB on a nonstationary bandit parameter grid."""

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
    rolling_mean,
    write_convergence_csv,
    write_summary_csv,
)
from experiments.bandits.nonstationary_gaussian_bandit import (
    NonstationaryGaussianBandit,
)
from rl_lib.algorithms.tabular.ucb_greedy_bandits import UCBGreedyBandits


def run_grid(
    *,
    k: int,
    exploration_constants: list[float],
    initial_value: float,
    constant_step_sizes: list[float],
    reward_std: float,
    drift_std: float,
    steps: int,
    runs: int,
    seed: int,
) -> list[BanditConditionResult]:
    """Evaluate every UCB constant and update rule using paired seeds."""

    def make_environment(run_seed: int) -> NonstationaryGaussianBandit:
        return NonstationaryGaussianBandit(
            k=k,
            reward_std=reward_std,
            drift_std=drift_std,
            seed=run_seed,
        )

    conditions: list[tuple[str, float | None]] = [("sample_average", None)]
    conditions.extend(("constant", step_size) for step_size in constant_step_sizes)
    results = []
    for exploration_constant in exploration_constants:
        for update_rule, step_size in conditions:

            def make_agent(
                run_seed: int,
                c: float = exploration_constant,
                alpha: float | None = step_size,
            ) -> UCBGreedyBandits:
                return UCBGreedyBandits(
                    k=k,
                    c=c,
                    initial_value=initial_value,
                    step_size=alpha,
                    seed=run_seed,
                )

            measurements = evaluate_bandit(
                environment_factory=make_environment,
                agent_factory=make_agent,
                runs=runs,
                steps=steps,
                seed=seed,
            )
            results.append(
                BanditConditionResult(
                    parameters={
                        "update_rule": update_rule,
                        "step_size": "1/N" if step_size is None else step_size,
                        "c": exploration_constant,
                        "initial_value": initial_value,
                        "drift_std": drift_std,
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
    smoothing_window: int,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> None:
    samples = rolling_mean(samples, smoothing_window)
    mean, lower, upper = mean_and_confidence_interval(samples)
    if lower_bound is not None:
        mean = np.maximum(mean, lower_bound)
        lower = np.maximum(lower, lower_bound)
    if upper_bound is not None:
        mean = np.minimum(mean, upper_bound)
        upper = np.minimum(upper, upper_bound)

    (line,) = axis.plot(steps, mean, label=label)
    axis.fill_between(steps, lower, upper, color=line.get_color(), alpha=0.1)


def write_grid_figure(
    path: Path,
    results: list[BanditConditionResult],
    *,
    exploration_constants: list[float],
    constant_step_sizes: list[float],
    drift_std: float,
    smoothing_window: int,
) -> None:
    """Plot update-rule curves in one metric column per UCB constant."""
    steps = np.arange(1, results[0].measurements.rewards.shape[1] + 1)
    figure, axes = plt.subplots(
        nrows=3,
        ncols=len(exploration_constants),
        figsize=(5 * len(exploration_constants), 11),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    conditions_per_constant = 1 + len(constant_step_sizes)
    for constant_index, exploration_constant in enumerate(exploration_constants):
        for condition_index in range(conditions_per_constant):
            result_index = constant_index * conditions_per_constant + condition_index
            result = results[result_index]
            step_size = result.parameters["step_size"]
            if result.parameters["update_rule"] == "sample_average":
                label = "Sample average (alpha=1/N)"
            else:
                label = f"Constant alpha={float(step_size):g}"
            measurements = result.measurements
            _plot_with_confidence(
                axes[0, constant_index],
                steps,
                measurements.instantaneous_regret,
                label,
                smoothing_window=smoothing_window,
                lower_bound=0.0,
            )
            _plot_with_confidence(
                axes[1, constant_index],
                steps,
                measurements.selected_optimal_action,
                label,
                smoothing_window=smoothing_window,
                lower_bound=0.0,
                upper_bound=1.0,
            )
            _plot_with_confidence(
                axes[2, constant_index],
                steps,
                measurements.estimation_mse,
                label,
                smoothing_window=smoothing_window,
                lower_bound=0.0,
            )

        axes[0, constant_index].set_title(f"UCB c={exploration_constant:g}")
        axes[2, constant_index].set_xlabel("Interaction step")

    axes[0, 0].set_ylabel("Mean pseudo-regret")
    axes[1, 0].set_ylabel("Optimal-action rate")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[2, 0].set_ylabel("Estimation MSE")
    for axis in axes.flat:
        axis.grid(alpha=0.25)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncols=conditions_per_constant,
    )
    figure.suptitle(
        "Standard UCB with cumulative counts on a nonstationary bandit "
        f"(drift std={drift_std:g}, rolling window={smoothing_window})",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.align_ylabels(axes[:, 0])
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
    )
    parser.add_argument("--initial-value", type=float, default=0.0)
    parser.add_argument(
        "--constant-step-sizes",
        type=float,
        nargs="+",
        default=[0.01, 0.1],
        metavar="ALPHA",
    )
    parser.add_argument("--reward-std", type=float, default=1.0)
    parser.add_argument("--drift-std", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--smoothing-window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/bandits/nonstationary_ucb"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.k < 1:
        raise ValueError("k must be at least 1")
    if len(set(args.exploration_constants)) != len(args.exploration_constants):
        raise ValueError("exploration_constants must be unique")
    if any(
        not np.isfinite(exploration_constant) or exploration_constant < 0
        for exploration_constant in args.exploration_constants
    ):
        raise ValueError("exploration_constants must be finite and non-negative")
    if not np.isfinite(args.initial_value):
        raise ValueError("initial_value must be finite")
    if len(set(args.constant_step_sizes)) != len(args.constant_step_sizes):
        raise ValueError("constant_step_sizes must be unique")
    if any(not 0.0 < step_size <= 1.0 for step_size in args.constant_step_sizes):
        raise ValueError(
            "constant_step_sizes must be between 0 exclusive and 1 inclusive"
        )
    if not np.isfinite(args.reward_std) or args.reward_std < 0:
        raise ValueError("reward_std must be finite and non-negative")
    if not np.isfinite(args.drift_std) or args.drift_std < 0:
        raise ValueError("drift_std must be finite and non-negative")
    if args.steps < 1:
        raise ValueError("steps must be at least 1")
    if args.runs < 1:
        raise ValueError("runs must be at least 1")
    if not 1 <= args.smoothing_window <= args.steps:
        raise ValueError("smoothing_window must be between 1 and steps")


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results = run_grid(
        k=args.k,
        exploration_constants=args.exploration_constants,
        initial_value=args.initial_value,
        constant_step_sizes=args.constant_step_sizes,
        reward_std=args.reward_std,
        drift_std=args.drift_std,
        steps=args.steps,
        runs=args.runs,
        seed=args.seed,
    )
    algorithm = "nonstationary_standard_ucb"
    write_convergence_csv(run_dir / "convergence.csv", algorithm, results)
    write_summary_csv(run_dir / "summary.csv", algorithm, results)
    write_grid_figure(
        run_dir / "nonstationary_ucb.png",
        results,
        exploration_constants=args.exploration_constants,
        constant_step_sizes=args.constant_step_sizes,
        drift_std=args.drift_std,
        smoothing_window=args.smoothing_window,
    )

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "experiment": "nonstationary_standard_ucb_parameter_grid",
        "k": args.k,
        "exploration_constants": args.exploration_constants,
        "initial_value": args.initial_value,
        "constant_step_sizes": args.constant_step_sizes,
        "reward_std": args.reward_std,
        "drift_std": args.drift_std,
        "steps": args.steps,
        "runs": args.runs,
        "smoothing_window": args.smoothing_window,
        "base_seed": args.seed,
        "count_weighting": "cumulative",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote nonstationary UCB experiment to {run_dir}")


if __name__ == "__main__":
    main()
