"""Compare implemented bandit algorithms on Gymnasium bandit environments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.bandits.environments import GaussianBandit
from experiments.common import create_run_directory, write_csv, write_metadata
from experiments.summary import SummaryMedia, SummaryTable, write_summary
from rl_lib.algorithms.bandits import EpsilonGreedyBandits, UCBGreedyBandits


@dataclass(frozen=True)
class BanditResult:
    environment: str
    algorithm: str
    parameter: float
    rewards: np.ndarray
    optimal_actions: np.ndarray


def run_condition(
    environment: str,
    algorithm: str,
    parameter: float,
    *,
    runs: int,
    steps: int,
    seed: int,
) -> BanditResult:
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

    return BanditResult(environment, algorithm, parameter, rewards, optimal)


def run_experiment(runs: int, steps: int, seed: int) -> list[BanditResult]:
    results = []
    for environment in ("stationary", "nonstationary"):
        for algorithm, parameters in (
            ("epsilon_greedy", (0.01, 0.1)),
            ("ucb", (1.0, 2.0)),
        ):
            for parameter in parameters:
                print(
                    f"Condition: {environment}, {algorithm}, {parameter:g}",
                    flush=True,
                )
                results.append(
                    run_condition(
                        environment,
                        algorithm,
                        parameter,
                        runs=runs,
                        steps=steps,
                        seed=seed,
                    )
                )
    return results


def _metric_rows(results: list[BanditResult]) -> list[dict[str, object]]:
    rows = []
    for result in results:
        runs, steps = result.rewards.shape
        for step in range(steps):
            rows.append(
                {
                    "environment": result.environment,
                    "algorithm": result.algorithm,
                    "parameter": result.parameter,
                    "step": step + 1,
                    "mean_reward": float(np.mean(result.rewards[:, step])),
                    "reward_std": (
                        float(np.std(result.rewards[:, step], ddof=1))
                        if runs > 1
                        else 0.0
                    ),
                    "optimal_action_rate": float(
                        np.mean(result.optimal_actions[:, step])
                    ),
                    "optimal_action_std": (
                        float(np.std(result.optimal_actions[:, step], ddof=1))
                        if runs > 1
                        else 0.0
                    ),
                }
            )
    return rows


def _late_summary(
    results: list[BanditResult],
    steps: int,
) -> list[tuple[object, ...]]:
    window = min(100, steps)
    summary = []
    for result in sorted(
        results,
        key=lambda item: (item.environment, item.algorithm, item.parameter),
    ):
        summary.append(
            (
                result.environment,
                result.algorithm,
                f"{result.parameter:g}",
                f"{np.mean(result.rewards[:, -window:]):.3f}",
                f"{np.mean(result.optimal_actions[:, -window:]):.3f}",
            )
        )
    return summary


def _rolling_mean(samples: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(samples, axis=1, dtype=float)
    smoothed = cumulative.copy()
    smoothed[:, window:] -= cumulative[:, :-window]
    counts = np.minimum(np.arange(1, samples.shape[1] + 1), window)
    return smoothed / counts


def _write_figure(
    path: Path,
    results: list[BanditResult],
    smoothing_window: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, environment in enumerate(("stationary", "nonstationary")):
        selected_results = sorted(
            (result for result in results if result.environment == environment),
            key=lambda item: (item.algorithm, item.parameter),
        )
        for result in selected_results:
            label = (
                f"epsilon-greedy, epsilon={result.parameter:g}"
                if result.algorithm == "epsilon_greedy"
                else f"UCB, c={result.parameter:g}"
            )
            for row_index, samples in enumerate(
                (result.rewards, result.optimal_actions.astype(float))
            ):
                smoothed = _rolling_mean(samples, smoothing_window)
                means = np.mean(smoothed, axis=0)
                standard_errors = (
                    np.std(smoothed, axis=0, ddof=1) / np.sqrt(smoothed.shape[0])
                    if smoothed.shape[0] > 1
                    else np.zeros(smoothed.shape[1])
                )
                steps = np.arange(1, samples.shape[1] + 1)
                axis = axes[row_index, column]
                (line,) = axis.plot(steps, means, label=label)
                axis.fill_between(
                    steps,
                    means - 1.96 * standard_errors,
                    means + 1.96 * standard_errors,
                    color=line.get_color(),
                    alpha=0.12,
                )
        axes[0, column].set_title(environment.capitalize())
        axes[1, column].set_xlabel("Interaction step")
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
            axis.legend()
    axes[0, 0].set_ylabel("Mean reward")
    axes[1, 0].set_ylabel("Optimal-action rate")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 1].set_ylim(-0.02, 1.02)
    figure.suptitle(f"Bandit learning curves ({smoothing_window}-step trailing mean)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_outputs(
    output: Path,
    results: list[BanditResult],
    *,
    runs: int,
    steps: int,
    seed: int,
    smoothing_window: int,
) -> None:
    figures = output / "figures"
    figures.mkdir()
    rows = _metric_rows(results)
    write_csv(output / "metrics.csv", tuple(rows[0]), rows)
    metadata = {
        "environment": "GaussianBandit",
        "runs": runs,
        "steps": steps,
        "seed": seed,
        "stationary_drift_std": 0.0,
        "nonstationary_drift_std": 0.01,
        "nonstationary_step_size": 0.1,
        "epsilon_values": [0.01, 0.1],
        "ucb_values": [1.0, 2.0],
        "confidence_interval": "mean +/- 1.96 * standard_error",
        "late_window_steps": min(100, steps),
        "smoothing_window": smoothing_window,
    }
    write_metadata(output / "metadata.json", metadata)
    figure_path = figures / "learning_curves.png"
    _write_figure(figure_path, results, smoothing_window)
    write_summary(
        output / "summary.html",
        title="Bandit experiment",
        metadata={
            "Environments": "Stationary and nonstationary Gaussian bandit",
            "Independent runs": runs,
            "Interactions per run": steps,
            "Smoothing window": f"{smoothing_window} steps",
        },
        tables=(
            SummaryTable(
                "Late-training metrics",
                (
                    "Environment",
                    "Algorithm",
                    "Parameter",
                    "Mean reward",
                    "Optimal-action rate",
                ),
                _late_summary(results, steps),
            ),
        ),
        figures=(
            SummaryMedia(
                "Reward and optimal-action learning curves",
                Path("figures/learning_curves.png"),
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--smoothing-window", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.runs < 1 or args.steps < 1 or args.smoothing_window < 1:
        parser.error("runs, steps, and smoothing window must be positive")

    output = create_run_directory("bandits")
    results = run_experiment(args.runs, args.steps, args.seed)
    write_outputs(
        output,
        results,
        runs=args.runs,
        steps=args.steps,
        seed=args.seed,
        smoothing_window=min(args.smoothing_window, args.steps),
    )
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
