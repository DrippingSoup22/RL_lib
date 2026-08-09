"""Train and compare on-policy Monte Carlo control on the looping MDP."""

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
from numpy.typing import NDArray

from experiments.monte_carlo.episodes import generate_episode
from experiments.monte_carlo.looping_mdp import LoopingMDP
from rl_lib.algorithms.tabular.monte_carlo_control import (
    EveryVisitMonteCarloControl,
    FirstVisitMonteCarloControl,
)
from rl_lib.algorithms.tabular.policies import policy_from_action_values

ENVIRONMENT_NAME = "looping_mdp"
GYM_ID = "custom/LoopingMDP-v0"

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
    "final_state",
    "success",
    "terminated",
    "truncated",
)

TRAINING_FIELDS = (
    "run_id",
    "algorithm",
    "seed",
    "episode",
    "final_state",
    "success",
    "epsilon",
    "q_start_action_0",
    "q_start_action_1",
    "q_decision_action_0",
    "q_decision_action_1",
    "probability_start_action_0",
    "probability_decision_action_1",
)

ControlClass = type[FirstVisitMonteCarloControl] | type[EveryVisitMonteCarloControl]


@dataclass(frozen=True)
class ControlTrainingResult:
    """Episode measurements and final tables from one seeded training run."""

    episode_records: list[dict[str, Any]]
    training_records: list[dict[str, Any]]
    action_values: NDArray[np.float64]
    visit_counts: NDArray[np.int_]
    policy_probabilities: NDArray[np.float64]


def train_control(
    *,
    run_id: str,
    algorithm: str,
    control_class: ControlClass,
    episodes: int,
    epsilon: float,
    discount: float,
    max_episode_steps: int,
    seed: int,
) -> ControlTrainingResult:
    """Train one control condition and record its per-episode progression."""
    _validate_training_parameters(
        episodes=episodes,
        epsilon=epsilon,
        discount=discount,
        max_episode_steps=max_episode_steps,
    )

    environment = gym.wrappers.TimeLimit(
        LoopingMDP(),
        max_episode_steps=max_episode_steps,
    )
    control = control_class(
        number_of_states=4,
        number_of_actions=2,
        epsilon=epsilon,
        discount=discount,
        seed=seed,
    )
    episode_records: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []

    try:
        for episode_index in range(episodes):
            episode = generate_episode(
                environment,
                control,
                seed=seed + episode_index,
            )
            episode_records.append(
                {
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "environment": ENVIRONMENT_NAME,
                    "gym_id": GYM_ID,
                    "seed": seed,
                    "phase": "training",
                    "episode": episode_index,
                    "return": sum(step.reward for step in episode.steps),
                    "length": len(episode.steps),
                    "final_state": episode.final_state,
                    "success": episode.final_state == LoopingMDP.SUCCESS,
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                }
            )

            control.update(episode)
            training_records.append(
                {
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": episode_index,
                    "final_state": episode.final_state,
                    "success": episode.final_state == LoopingMDP.SUCCESS,
                    "epsilon": epsilon,
                    "q_start_action_0": control.action_values[LoopingMDP.START, 0],
                    "q_start_action_1": control.action_values[LoopingMDP.START, 1],
                    "q_decision_action_0": control.action_values[
                        LoopingMDP.DECISION, 0
                    ],
                    "q_decision_action_1": control.action_values[
                        LoopingMDP.DECISION, 1
                    ],
                    "probability_start_action_0": control.policy.probabilities[
                        LoopingMDP.START, 0
                    ],
                    "probability_decision_action_1": control.policy.probabilities[
                        LoopingMDP.DECISION, 1
                    ],
                }
            )
    finally:
        environment.close()

    return ControlTrainingResult(
        episode_records=episode_records,
        training_records=training_records,
        action_values=control.action_values.copy(),
        visit_counts=control.visit_counts.copy(),
        policy_probabilities=control.policy.probabilities.copy(),
    )


def evaluate_action_values(
    *,
    run_id: str,
    algorithm: str,
    action_values: NDArray[np.float64],
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate the greedy policy implied by a learned action-value table."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be at least 1")

    policy = policy_from_action_values(action_values, epsilon=0.0, seed=seed)
    environment = gym.wrappers.TimeLimit(
        LoopingMDP(),
        max_episode_steps=max_episode_steps,
    )
    records: list[dict[str, Any]] = []

    try:
        for episode_index in range(episodes):
            episode = generate_episode(
                environment,
                policy,
                seed=seed + episode_index,
            )
            records.append(
                {
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "environment": ENVIRONMENT_NAME,
                    "gym_id": GYM_ID,
                    "seed": seed,
                    "phase": "evaluation",
                    "episode": episode_index,
                    "return": sum(step.reward for step in episode.steps),
                    "length": len(episode.steps),
                    "final_state": episode.final_state,
                    "success": episode.final_state == LoopingMDP.SUCCESS,
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                }
            )
    finally:
        environment.close()

    return records


def _validate_training_parameters(
    *,
    episodes: int,
    epsilon: float,
    discount: float,
    max_episode_steps: int,
) -> None:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if not np.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be finite and between 0 exclusive and 1")
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be finite and between 0 and 1")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be at least 1")


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
    training_records: list[dict[str, Any]],
    final_estimates: list[dict[str, Any]],
    episodes_per_seed: int,
    evaluation_episodes_per_seed: int,
    seeds: list[int],
    epsilon: float,
    discount: float,
    max_episode_steps: int,
) -> str:
    """Render a human-readable summary of one control comparison."""
    late_window = min(100, episodes_per_seed)
    late_start = episodes_per_seed - late_window
    algorithm_names = {
        "first_visit_mc_control": "First-visit MC control",
        "every_visit_mc_control": "Every-visit MC control",
    }
    algorithms = tuple(algorithm_names)

    lines = [
        "# Monte Carlo control results",
        "",
        f"Source run: `{run_id}`",
        "",
        "This report compares first-visit and every-visit on-policy Monte Carlo",
        "control on the deterministic looping MDP. Both methods learn from normal",
        "resets using an epsilon-soft policy; exploring starts are not used.",
        "",
        "## Configuration",
        "",
        f"- Seeds: `{seeds}`",
        f"- Training episodes per seed: `{episodes_per_seed}`",
        f"- Greedy evaluation episodes per seed: `{evaluation_episodes_per_seed}`",
        f"- Epsilon: `{epsilon:g}`",
        f"- Discount: `{discount:g}`",
        f"- Maximum episode length: `{max_episode_steps}`",
        f"- Late-training window: final `{late_window}` episodes per seed",
        "",
        "## Late-training performance",
        "",
        "These measurements come from the epsilon-soft training policy, so they",
        "include deliberate exploratory actions.",
        "",
        "| Algorithm | Mean return | Success rate | Mean length | Truncation rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for algorithm in algorithms:
        samples = [
            record
            for record in episode_records
            if record["algorithm"] == algorithm
            and record["phase"] == "training"
            and record["episode"] >= late_start
        ]
        training_samples = [
            record
            for record in training_records
            if record["algorithm"] == algorithm and record["episode"] >= late_start
        ]
        mean_return = float(np.mean([record["return"] for record in samples]))
        success_rate = float(
            np.mean([record["success"] for record in training_samples])
        )
        mean_length = float(np.mean([record["length"] for record in samples]))
        truncation_rate = float(np.mean([record["truncated"] for record in samples]))
        lines.append(
            f"| {algorithm_names[algorithm]} | {mean_return:.3f} | "
            f"{success_rate:.3f} | {mean_length:.3f} | {truncation_rate:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Greedy evaluation performance",
            "",
            "These episodes use a frozen policy constructed from the final Q table",
            "with epsilon=0. No learning updates occur during evaluation.",
            "",
            "| Algorithm | Mean return | Success rate | Mean length | "
            "Truncation rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for algorithm in algorithms:
        samples = [
            record
            for record in episode_records
            if record["algorithm"] == algorithm and record["phase"] == "evaluation"
        ]
        mean_return = float(np.mean([record["return"] for record in samples]))
        success_rate = float(np.mean([record["success"] for record in samples]))
        mean_length = float(np.mean([record["length"] for record in samples]))
        truncation_rate = float(np.mean([record["truncated"] for record in samples]))
        lines.append(
            f"| {algorithm_names[algorithm]} | {mean_return:.3f} | "
            f"{success_rate:.3f} | {mean_length:.3f} | {truncation_rate:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Final learned estimates",
            "",
            "Each table averages the final learned arrays across seeds. State 0 is",
            "START and state 1 is DECISION. The desired choices are action 0 at",
            "START and action 1 at DECISION.",
        ]
    )

    for algorithm in algorithms:
        estimates = [
            estimate
            for estimate in final_estimates
            if estimate["algorithm"] == algorithm
        ]
        mean_q = np.mean(
            [
                np.asarray(estimate["action_values"], dtype=float)
                for estimate in estimates
            ],
            axis=0,
        )
        mean_policy = np.mean(
            [
                np.asarray(estimate["policy_probabilities"], dtype=float)
                for estimate in estimates
            ],
            axis=0,
        )
        lines.extend(
            [
                "",
                f"### {algorithm_names[algorithm]}",
                "",
                "| State | Q(action 0) | Q(action 1) | P(action 0) | P(action 1) |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| START | {mean_q[LoopingMDP.START, 0]:.3f} | "
                f"{mean_q[LoopingMDP.START, 1]:.3f} | "
                f"{mean_policy[LoopingMDP.START, 0]:.3f} | "
                f"{mean_policy[LoopingMDP.START, 1]:.3f} |",
                f"| DECISION | {mean_q[LoopingMDP.DECISION, 0]:.3f} | "
                f"{mean_q[LoopingMDP.DECISION, 1]:.3f} | "
                f"{mean_policy[LoopingMDP.DECISION, 0]:.3f} | "
                f"{mean_policy[LoopingMDP.DECISION, 1]:.3f} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Both algorithms should prefer action 0 at START, avoiding immediate",
            "failure, and action 1 at DECISION, reaching success instead of looping.",
            "With two actions and epsilon=0.1, a uniquely greedy action receives",
            "probability 0.95 while the other retains probability 0.05.",
            "",
            "First-visit and every-visit methods can assign different transient",
            "estimates to the looping action because every-visit uses all repeated",
            "occurrences in an episode. Agreement on the final preferred actions is",
            "the principal control result.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/monte_carlo/control"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_training_parameters(
        episodes=args.episodes,
        epsilon=args.epsilon,
        discount=args.discount,
        max_episode_steps=args.max_episode_steps,
    )
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if args.evaluation_episodes < 1:
        raise ValueError("evaluation_episodes must be at least 1")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    conditions: tuple[tuple[str, ControlClass], ...] = (
        ("first_visit_mc_control", FirstVisitMonteCarloControl),
        ("every_visit_mc_control", EveryVisitMonteCarloControl),
    )
    all_episode_records: list[dict[str, Any]] = []
    all_training_records: list[dict[str, Any]] = []
    final_estimates: list[dict[str, Any]] = []

    for algorithm, control_class in conditions:
        for seed in args.seeds:
            result = train_control(
                run_id=run_id,
                algorithm=algorithm,
                control_class=control_class,
                episodes=args.episodes,
                epsilon=args.epsilon,
                discount=args.discount,
                max_episode_steps=args.max_episode_steps,
                seed=seed,
            )
            all_episode_records.extend(result.episode_records)
            all_episode_records.extend(
                evaluate_action_values(
                    run_id=run_id,
                    algorithm=algorithm,
                    action_values=result.action_values,
                    episodes=args.evaluation_episodes,
                    max_episode_steps=args.max_episode_steps,
                    seed=seed,
                )
            )
            all_training_records.extend(result.training_records)
            final_estimates.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "action_values": result.action_values.tolist(),
                    "visit_counts": result.visit_counts.tolist(),
                    "policy_probabilities": result.policy_probabilities.tolist(),
                }
            )

    _write_csv(run_dir / "episodes.csv", EPISODE_FIELDS, all_episode_records)
    _write_csv(run_dir / "training.csv", TRAINING_FIELDS, all_training_records)
    (run_dir / "final_estimates.json").write_text(
        json.dumps(final_estimates, indent=2) + "\n",
        encoding="utf-8",
    )
    report = render_report(
        run_id=run_id,
        episode_records=all_episode_records,
        training_records=all_training_records,
        final_estimates=final_estimates,
        episodes_per_seed=args.episodes,
        evaluation_episodes_per_seed=args.evaluation_episodes,
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
        "experiment": "looping_mdp_monte_carlo_control",
        "algorithms": [algorithm for algorithm, _ in conditions],
        "environment": ENVIRONMENT_NAME,
        "gym_id": GYM_ID,
        "episodes_per_seed": args.episodes,
        "evaluation_episodes_per_seed": args.evaluation_episodes,
        "seeds": args.seeds,
        "epsilon": args.epsilon,
        "discount": args.discount,
        "max_episode_steps": args.max_episode_steps,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Monte Carlo control experiment to {run_dir}")


if __name__ == "__main__":
    main()
