"""Compare Monte Carlo prediction methods for one frozen learned policy."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray

from experiments.monte_carlo.control import train_control
from experiments.monte_carlo.episodes import generate_episode
from experiments.monte_carlo.looping_mdp import LoopingMDP
from rl_lib.algorithms.tabular.monte_carlo_control import (
    FirstVisitMonteCarloControl,
)
from rl_lib.algorithms.tabular.monte_carlo_prediction import (
    EveryVisitMonteCarloPrediction,
    FirstVisitMonteCarloPrediction,
)
from rl_lib.algorithms.tabular.policies import policy_from_action_values

ENVIRONMENT_NAME = "looping_mdp"
GYM_ID = "custom/LoopingMDP-v0"

EPISODE_FIELDS = (
    "run_id",
    "environment",
    "gym_id",
    "seed",
    "phase",
    "episode",
    "return",
    "length",
    "final_state",
    "success",
    "terminated",
    "truncated",
)

TRACE_FIELDS = (
    "run_id",
    "method",
    "seed",
    "episode",
    "value_start",
    "value_decision",
    "visits_start",
    "visits_decision",
    "absolute_error_start",
    "absolute_error_decision",
)

METHOD_NAMES = {
    "first_visit_mc_prediction": "First-visit MC prediction",
    "every_visit_mc_prediction": "Every-visit MC prediction",
}


@dataclass(frozen=True)
class PredictionComparisonResult:
    """Measurements and final estimates for one seeded comparison."""

    episode_records: list[dict[str, Any]]
    trace_records: list[dict[str, Any]]
    policy_probabilities: NDArray[np.float64]
    true_values: NDArray[np.float64]
    final_values: dict[str, NDArray[np.float64]]
    visit_counts: dict[str, NDArray[np.int_]]


def true_looping_mdp_values(
    policy_probabilities: ArrayLike,
    discount: float,
) -> NDArray[np.float64]:
    """Calculate exact state values for a policy on ``LoopingMDP``."""
    probabilities = np.asarray(policy_probabilities, dtype=float)
    if probabilities.shape != (4, 2):
        raise ValueError("policy_probabilities must have shape (4, 2)")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("policy_probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("policy_probabilities must be between 0 and 1")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0):
        raise ValueError("each policy row must sum to 1")
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be finite and between 0 and 1")

    loop_probability = probabilities[LoopingMDP.DECISION, 0]
    success_probability = probabilities[LoopingMDP.DECISION, 1]
    denominator = 1.0 - discount * loop_probability
    if np.isclose(denominator, 0.0):
        raise ValueError("policy does not terminate from the decision state")

    values = np.zeros(4, dtype=float)
    values[LoopingMDP.DECISION] = (
        loop_probability * -0.1 + success_probability
    ) / denominator
    values[LoopingMDP.START] = (
        probabilities[LoopingMDP.START, 0] * discount * values[LoopingMDP.DECISION]
        - probabilities[LoopingMDP.START, 1]
    )
    return values


def compare_prediction_methods(
    *,
    run_id: str,
    action_values: ArrayLike,
    episodes: int,
    epsilon: float,
    discount: float,
    max_episode_steps: int,
    seed: int,
) -> PredictionComparisonResult:
    """Evaluate one frozen policy with both visit conventions."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be at least 1")

    policy = policy_from_action_values(action_values, epsilon=epsilon, seed=seed)
    frozen_probabilities = policy.probabilities.copy()
    true_values = true_looping_mdp_values(frozen_probabilities, discount)
    predictors = {
        "first_visit_mc_prediction": FirstVisitMonteCarloPrediction(4, discount),
        "every_visit_mc_prediction": EveryVisitMonteCarloPrediction(4, discount),
    }
    environment = gym.wrappers.TimeLimit(
        LoopingMDP(),
        max_episode_steps=max_episode_steps,
    )
    episode_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []

    try:
        for episode_index in range(episodes):
            episode = generate_episode(
                environment,
                policy,
                seed=seed + episode_index,
            )
            episode_records.append(
                {
                    "run_id": run_id,
                    "environment": ENVIRONMENT_NAME,
                    "gym_id": GYM_ID,
                    "seed": seed,
                    "phase": "prediction",
                    "episode": episode_index,
                    "return": sum(step.reward for step in episode.steps),
                    "length": len(episode.steps),
                    "final_state": episode.final_state,
                    "success": episode.final_state == LoopingMDP.SUCCESS,
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                }
            )

            for method, predictor in predictors.items():
                predictor.update(episode)
                trace_records.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "seed": seed,
                        "episode": episode_index,
                        "value_start": predictor.values[LoopingMDP.START],
                        "value_decision": predictor.values[LoopingMDP.DECISION],
                        "visits_start": predictor.visit_counts[LoopingMDP.START],
                        "visits_decision": predictor.visit_counts[LoopingMDP.DECISION],
                        "absolute_error_start": abs(
                            predictor.values[LoopingMDP.START]
                            - true_values[LoopingMDP.START]
                        ),
                        "absolute_error_decision": abs(
                            predictor.values[LoopingMDP.DECISION]
                            - true_values[LoopingMDP.DECISION]
                        ),
                    }
                )
    finally:
        environment.close()

    if not np.array_equal(policy.probabilities, frozen_probabilities):
        raise RuntimeError("prediction changed the frozen policy")
    return PredictionComparisonResult(
        episode_records=episode_records,
        trace_records=trace_records,
        policy_probabilities=frozen_probabilities,
        true_values=true_values,
        final_values={
            method: predictor.values.copy() for method, predictor in predictors.items()
        },
        visit_counts={
            method: predictor.visit_counts.copy()
            for method, predictor in predictors.items()
        },
    )


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    records: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def render_report(
    *,
    run_id: str,
    episode_records: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
    final_estimates: list[dict[str, Any]],
    control_episodes: int,
    prediction_episodes: int,
    seeds: list[int],
    epsilon: float,
    discount: float,
    max_episode_steps: int,
) -> str:
    """Render a human-readable summary of the prediction comparison."""
    lines = [
        "# Monte Carlo prediction results",
        "",
        f"Source run: `{run_id}`",
        "",
        "This experiment evaluates one frozen epsilon-soft policy with both",
        "first-visit and every-visit Monte Carlo prediction. Each pair of",
        "estimators receives exactly the same generated episodes.",
        "",
        "## Configuration",
        "",
        f"- Seeds: `{seeds}`",
        f"- Control-training episodes per seed: `{control_episodes}`",
        f"- Prediction episodes per seed: `{prediction_episodes}`",
        f"- Frozen-policy epsilon: `{epsilon:g}`",
        f"- Discount: `{discount:g}`",
        f"- Maximum episode length: `{max_episode_steps}`",
        "",
        "## Frozen policy and exact values",
        "",
        "The policy is constructed from the Q table learned by first-visit MC",
        "control and is not changed during prediction. Values are calculated",
        "analytically from this policy before sampling begins.",
        "",
        "| State | P(action 0) | P(action 1) | Exact V_pi |",
        "| --- | ---: | ---: | ---: |",
    ]

    mean_policy = np.mean(
        [
            np.asarray(item["policy_probabilities"], dtype=float)
            for item in final_estimates
        ],
        axis=0,
    )
    mean_true_values = np.mean(
        [np.asarray(item["true_values"], dtype=float) for item in final_estimates],
        axis=0,
    )
    for state, name in (
        (LoopingMDP.START, "START"),
        (LoopingMDP.DECISION, "DECISION"),
    ):
        lines.append(
            f"| {name} | {mean_policy[state, 0]:.3f} | "
            f"{mean_policy[state, 1]:.3f} | {mean_true_values[state]:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Final prediction estimates",
            "",
            "Errors are absolute differences from the exact value of each seeded",
            "frozen policy.",
            "",
            "| Method | V(START) | V(DECISION) | Mean absolute error | "
            "DECISION visits |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, display_name in METHOD_NAMES.items():
        samples = [
            record
            for record in trace_records
            if record["method"] == method
            and record["episode"] == prediction_episodes - 1
        ]
        mean_error = float(
            np.mean(
                [
                    (record["absolute_error_start"] + record["absolute_error_decision"])
                    / 2
                    for record in samples
                ]
            )
        )
        lines.append(
            f"| {display_name} | "
            f"{np.mean([record['value_start'] for record in samples]):.3f} | "
            f"{np.mean([record['value_decision'] for record in samples]):.3f} | "
            f"{mean_error:.3f} | "
            f"{np.mean([record['visits_decision'] for record in samples]):.1f} |"
        )

    checkpoints = sorted(
        {
            0,
            min(9, prediction_episodes - 1),
            min(99, prediction_episodes - 1),
            prediction_episodes - 1,
        }
    )
    lines.extend(
        [
            "",
            "## Error during learning",
            "",
            "| Episodes | First-visit MAE | Every-visit MAE |",
            "| ---: | ---: | ---: |",
        ]
    )
    for checkpoint in checkpoints:
        errors = {}
        for method in METHOD_NAMES:
            samples = [
                record
                for record in trace_records
                if record["method"] == method and record["episode"] == checkpoint
            ]
            errors[method] = float(
                np.mean(
                    [
                        (
                            record["absolute_error_start"]
                            + record["absolute_error_decision"]
                        )
                        / 2
                        for record in samples
                    ]
                )
            )
        lines.append(
            f"| {checkpoint + 1} | "
            f"{errors['first_visit_mc_prediction']:.3f} | "
            f"{errors['every_visit_mc_prediction']:.3f} |"
        )

    mean_return = float(np.mean([record["return"] for record in episode_records]))
    mean_length = float(np.mean([record["length"] for record in episode_records]))
    truncation_rate = float(
        np.mean([record["truncated"] for record in episode_records])
    )
    lines.extend(
        [
            "",
            "## Episode sample",
            "",
            f"- Mean undiscounted return: `{mean_return:.3f}`",
            f"- Mean episode length: `{mean_length:.3f}`",
            f"- Truncation rate: `{truncation_rate:.3f}`",
            "",
            "## Interpretation",
            "",
            "Both methods estimate the value function of the same fixed policy.",
            "Every-visit prediction processes every repeated occurrence of a state,",
            "so its DECISION visit count is larger. First-visit prediction uses at",
            "most one return per state per episode. Their transient estimates and",
            "sample counts can differ even though both target the same V_pi.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-episodes", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/monte_carlo/prediction"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.control_episodes < 1:
        raise ValueError("control_episodes must be at least 1")
    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    all_episode_records: list[dict[str, Any]] = []
    all_trace_records: list[dict[str, Any]] = []
    final_estimates: list[dict[str, Any]] = []

    for seed in args.seeds:
        control_result = train_control(
            run_id=run_id,
            algorithm="first_visit_mc_control",
            control_class=FirstVisitMonteCarloControl,
            episodes=args.control_episodes,
            epsilon=args.epsilon,
            discount=args.discount,
            max_episode_steps=args.max_episode_steps,
            seed=seed,
        )
        result = compare_prediction_methods(
            run_id=run_id,
            action_values=control_result.action_values,
            episodes=args.episodes,
            epsilon=args.epsilon,
            discount=args.discount,
            max_episode_steps=args.max_episode_steps,
            seed=seed,
        )
        all_episode_records.extend(result.episode_records)
        all_trace_records.extend(result.trace_records)
        final_estimates.append(
            {
                "seed": seed,
                "source_action_values": control_result.action_values.tolist(),
                "policy_probabilities": result.policy_probabilities.tolist(),
                "true_values": result.true_values.tolist(),
                "final_values": {
                    method: values.tolist()
                    for method, values in result.final_values.items()
                },
                "visit_counts": {
                    method: counts.tolist()
                    for method, counts in result.visit_counts.items()
                },
            }
        )

    _write_csv(run_dir / "episodes.csv", EPISODE_FIELDS, all_episode_records)
    _write_csv(run_dir / "prediction.csv", TRACE_FIELDS, all_trace_records)
    (run_dir / "final_estimates.json").write_text(
        json.dumps(final_estimates, indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(
        run_id=run_id,
        episode_records=all_episode_records,
        trace_records=all_trace_records,
        final_estimates=final_estimates,
        control_episodes=args.control_episodes,
        prediction_episodes=args.episodes,
        seeds=args.seeds,
        epsilon=args.epsilon,
        discount=args.discount,
        max_episode_steps=args.max_episode_steps,
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "experiment": "looping_mdp_monte_carlo_prediction",
        "methods": list(METHOD_NAMES),
        "policy_source": "first_visit_mc_control",
        "environment": ENVIRONMENT_NAME,
        "gym_id": GYM_ID,
        "control_episodes_per_seed": args.control_episodes,
        "prediction_episodes_per_seed": args.episodes,
        "seeds": args.seeds,
        "epsilon": args.epsilon,
        "discount": args.discount,
        "max_episode_steps": args.max_episode_steps,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Monte Carlo prediction experiment to {run_dir}")


if __name__ == "__main__":
    main()
