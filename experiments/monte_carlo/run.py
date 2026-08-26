"""Run Monte Carlo prediction and control on tabular Gymnasium environments."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from experiments.common import (
    RECORDING_CHECKPOINTS,
    annotated_frame,
    checkpoint_episode_target,
    create_run_directory,
    evaluation_checkpoints,
    resolve_seed_values,
    write_csv,
    write_metadata,
)
from experiments.monte_carlo.environments import BLACKJACK_ACTIONS as BLACKJACK_ACTIONS
from experiments.monte_carlo.environments import BLACKJACK_STATES as BLACKJACK_STATES
from experiments.monte_carlo.environments import (
    KNOWN_ENVIRONMENTS,
    action_name,
    encode_blackjack_state,
    encode_discrete_state,
    episode_succeeded,
    experiment_defaults,
    fixed_prediction_action,
    inspect_environment,
    make_environment,
    recording_settings,
    success_definition,
)
from experiments.monte_carlo.report import (
    write_control_report,
    write_prediction_report,
)
from rl_lib.algorithms.monte_carlo import (
    EveryVisitMonteCarloControl,
    EveryVisitMonteCarloPrediction,
    FirstVisitMonteCarloControl,
    FirstVisitMonteCarloPrediction,
)
from rl_lib.data.episode import Episode, EpisodeStep

ALGORITHMS = (
    "first_visit_prediction",
    "every_visit_prediction",
    "first_visit_control",
    "every_visit_control",
)


def generate_episode(
    env: gym.Env,
    select_action: Callable[[int], int],
    *,
    seed: int,
    encode_observation: Callable[[Any], int] = encode_discrete_state,
) -> Episode[int]:
    observation, _ = env.reset(seed=seed)
    state = encode_observation(observation)
    steps: list[EpisodeStep[int]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = select_action(state)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        steps.append(EpisodeStep(state, action, float(reward)))
        state = encode_observation(next_observation)
    return Episode(tuple(steps), state, terminated, truncated)


def greedy_action(
    action_values: np.ndarray,
    rng: np.random.Generator,
    state: int,
) -> int:
    values = action_values[state]
    return int(rng.choice(np.flatnonzero(values == np.max(values))))


def prediction_states(environment: str, number_of_states: int) -> list[int]:
    if environment != "Blackjack-v1":
        return list(range(number_of_states))
    return [
        encode_blackjack_state((player, dealer, usable_ace))
        for player in range(12, 22)
        for dealer in range(1, 11)
        for usable_ace in (0, 1)
    ]


def run_prediction(
    algorithm: str,
    environment: str,
    number_of_states: int,
    number_of_actions: int,
    encode_observation: Callable[[Any], int],
    episodes: int,
    seed: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    estimator = (
        FirstVisitMonteCarloPrediction(number_of_states)
        if algorithm == "first_visit_prediction"
        else EveryVisitMonteCarloPrediction(number_of_states)
    )
    rng = np.random.default_rng(seed)
    select_action = partial(
        fixed_prediction_action,
        environment,
        rng,
        number_of_actions=number_of_actions,
    )
    env = make_environment(environment)
    for episode in range(episodes):
        trajectory = generate_episode(
            env,
            select_action,
            seed=seed * episodes + episode,
            encode_observation=encode_observation,
        )
        estimator.update(trajectory)
    env.close()

    rows = [
        {
            "phase": "prediction",
            "algorithm": algorithm,
            "seed": seed,
            "checkpoint": 100,
            "state": state,
            "value": float(estimator.V[state]),
            "visits": int(estimator.visit_counts[state]),
            "mean_return": "",
            "return_std": "",
            "success_rate": "",
            "success_rate_std": "",
            "mean_episode_length": "",
            "episode_length_std": "",
            "truncation_rate": "",
            "truncation_rate_std": "",
            "illegal_actions": "",
            "illegal_actions_std": "",
            "draw_rate": "",
            "loss_rate": "",
            "evaluation_episodes": "",
        }
        for state in prediction_states(environment, number_of_states)
    ]
    return rows, estimator.V.copy()


def record_evaluation(
    path: Path,
    *,
    environment: str,
    encode_observation: Callable[[Any], int],
    action_values: np.ndarray,
    seeds: tuple[int, ...],
    frame_duration_ms: int,
    terminal_duration_ms: int,
) -> None:
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_environment(environment, render_mode="rgb_array")
    policy = partial(greedy_action, action_values, np.random.default_rng(0))
    frames = []
    durations = []
    for episode, seed in enumerate(seeds, start=1):
        observation, _ = env.reset(seed=seed)
        frames.append(annotated_frame(env.render(), f"Episode {episode}: start"))
        durations.append(frame_duration_ms)
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            action = policy(encode_observation(observation))
            observation, reward, terminated, truncated, _ = env.step(action)
            step += 1
            action_label = action_name(environment, action)
            frames.append(
                annotated_frame(
                    env.render(),
                    f"Episode {episode}, step {step}: action {action_label}, "
                    f"reward {float(reward):g}",
                )
            )
            durations.append(
                terminal_duration_ms if terminated or truncated else frame_duration_ms
            )
    env.close()
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


def _aggregate(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _control_metric_row(
    algorithm: str,
    checkpoint: int,
    checkpoint_metrics: dict[str, dict[int, list[float]]],
    evaluation_episodes: int,
    completed_seeds: int,
) -> dict[str, object]:
    mean_return, return_std = _aggregate(checkpoint_metrics["return"][checkpoint])
    success_rate, success_rate_std = _aggregate(
        checkpoint_metrics["success"][checkpoint]
    )
    mean_length, length_std = _aggregate(checkpoint_metrics["length"][checkpoint])
    truncation_rate, truncation_std = _aggregate(
        checkpoint_metrics["truncation"][checkpoint]
    )
    illegal_actions, illegal_actions_std = _aggregate(
        checkpoint_metrics["illegal_actions"][checkpoint]
    )
    draw_rate, _ = _aggregate(checkpoint_metrics["draw"][checkpoint])
    loss_rate, _ = _aggregate(checkpoint_metrics["loss"][checkpoint])
    return {
        "phase": "control",
        "algorithm": algorithm,
        "seed": "",
        "checkpoint": checkpoint,
        "state": "",
        "value": "",
        "visits": "",
        "mean_return": mean_return,
        "return_std": return_std,
        "success_rate": success_rate,
        "success_rate_std": success_rate_std,
        "mean_episode_length": mean_length,
        "episode_length_std": length_std,
        "truncation_rate": truncation_rate,
        "truncation_rate_std": truncation_std,
        "illegal_actions": illegal_actions,
        "illegal_actions_std": illegal_actions_std,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "evaluation_episodes": evaluation_episodes * completed_seeds,
    }


def _control_seed_row(
    algorithm: str,
    seed: int,
    checkpoint: int,
    samples: dict[str, list[float]],
    evaluation_episodes: int,
) -> dict[str, object]:
    """Keep one evaluated mean per seed instead of discarding it after aggregation."""
    return {
        "phase": "control_seed",
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint": checkpoint,
        "state": "",
        "value": "",
        "visits": "",
        "mean_return": float(np.mean(samples["return"])),
        "return_std": "",
        "success_rate": (
            float(np.mean(samples["success"])) if samples["success"] else ""
        ),
        "success_rate_std": "",
        "mean_episode_length": float(np.mean(samples["length"])),
        "episode_length_std": "",
        "truncation_rate": float(np.mean(samples["truncation"])),
        "truncation_rate_std": "",
        "illegal_actions": (
            float(np.mean(samples["illegal_actions"]))
            if samples["illegal_actions"]
            else ""
        ),
        "illegal_actions_std": "",
        "draw_rate": float(np.mean(samples["draw"])) if samples["draw"] else "",
        "loss_rate": float(np.mean(samples["loss"])) if samples["loss"] else "",
        "evaluation_episodes": evaluation_episodes,
    }


def run_control(
    algorithm: str,
    environment: str,
    number_of_states: int,
    number_of_actions: int,
    encode_observation: Callable[[Any], int],
    renderable: bool,
    training_episodes: int,
    evaluation_episodes: int,
    epsilon: float,
    seed_values: tuple[int, ...],
    recordings_directory: Path | None,
    recording_checkpoints: tuple[int, ...],
    checkpoints: tuple[int, ...],
    metrics_path: Path | None,
    initial_rows: list[dict[str, object]],
    model_path: Path | None,
) -> tuple[list[dict[str, object]], int]:
    rows = initial_rows.copy()
    control_class = (
        FirstVisitMonteCarloControl
        if algorithm == "first_visit_control"
        else EveryVisitMonteCarloControl
    )
    recording_values: dict[int, dict[int, np.ndarray]] = {}
    aggregate_rows: dict[int, dict[str, object]] = {}
    checkpoint_metrics: dict[str, dict[int, list[float]]] = {
        metric: {checkpoint: [] for checkpoint in checkpoints}
        for metric in (
            "return",
            "success",
            "length",
            "truncation",
            "illegal_actions",
            "draw",
            "loss",
        )
    }
    for trial_index, seed in enumerate(seed_values):
        print(
            f"Control: {algorithm}, seed trial {trial_index + 1}/{len(seed_values)} "
            f"(seed={seed})",
            flush=True,
        )
        env = make_environment(environment)
        agent = control_class(
            number_of_states,
            number_of_actions,
            epsilon=epsilon,
            seed=seed,
        )
        completed_episodes = 0
        for checkpoint in checkpoints:
            target = checkpoint_episode_target(training_episodes, checkpoint)
            print(
                f"  checkpoint {checkpoint}% ({target} training episodes)",
                flush=True,
            )
            while completed_episodes < target:
                trajectory = generate_episode(
                    env,
                    agent.select_action,
                    seed=seed * training_episodes + completed_episodes,
                    encode_observation=encode_observation,
                )
                agent.update(trajectory)
                completed_episodes += 1

            evaluation_policy = partial(
                greedy_action,
                agent.Q,
                np.random.default_rng(seed + checkpoint),
            )
            samples: dict[str, list[float]] = {
                "return": [],
                "success": [],
                "length": [],
                "truncation": [],
                "illegal_actions": [],
                "draw": [],
                "loss": [],
            }
            for episode in range(evaluation_episodes):
                trajectory = generate_episode(
                    env,
                    evaluation_policy,
                    seed=(
                        1_000_000
                        + checkpoint * 100_000
                        + seed * evaluation_episodes
                        + episode
                    ),
                    encode_observation=encode_observation,
                )
                episode_return = sum(step.reward for step in trajectory.steps)
                samples["return"].append(episode_return)
                samples["length"].append(float(len(trajectory.steps)))
                samples["truncation"].append(float(trajectory.truncated))
                succeeded = episode_succeeded(
                    environment,
                    terminated=trajectory.terminated,
                    final_reward=trajectory.steps[-1].reward,
                )
                if succeeded is not None:
                    samples["success"].append(float(succeeded))
                if environment == "Taxi-v4":
                    samples["illegal_actions"].append(
                        float(sum(step.reward == -10 for step in trajectory.steps))
                    )
                if environment == "Blackjack-v1":
                    samples["draw"].append(float(episode_return == 0))
                    samples["loss"].append(float(episode_return < 0))

            for metric, values in samples.items():
                if values:
                    checkpoint_metrics[metric][checkpoint].append(
                        float(np.mean(values))
                    )
            rows.append(
                _control_seed_row(
                    algorithm,
                    seed,
                    checkpoint,
                    samples,
                    evaluation_episodes,
                )
            )
            aggregate_rows[checkpoint] = _control_metric_row(
                algorithm,
                checkpoint,
                checkpoint_metrics,
                evaluation_episodes,
                trial_index + 1,
            )
            completed_rows = rows + [
                aggregate_rows[item] for item in checkpoints if item in aggregate_rows
            ]
            if metrics_path is not None:
                write_csv(metrics_path, tuple(rows[0]), completed_rows)

            if (renderable and checkpoint in recording_checkpoints) or (
                model_path is not None and checkpoint == 100
            ):
                recording_values.setdefault(seed, {})[checkpoint] = agent.Q.copy()
        env.close()
    rows.extend(aggregate_rows[checkpoint] for checkpoint in checkpoints)
    if metrics_path is not None:
        write_csv(metrics_path, tuple(rows[0]), rows)

    final_seed_rows = [
        row
        for row in rows
        if row["phase"] == "control_seed" and row["checkpoint"] == 100
    ]
    selected_row = max(
        final_seed_rows,
        key=lambda row: (
            float(row["mean_return"]),
            float(row["success_rate"]) if row["success_rate"] != "" else 0.0,
        ),
    )
    selected_seed = int(selected_row["seed"])
    if model_path is not None:
        np.savez_compressed(
            model_path,
            algorithm=np.asarray(algorithm),
            seed=np.asarray(selected_seed),
            checkpoint=np.asarray(100),
            Q=recording_values[selected_seed][100],
        )
    if renderable and recording_checkpoints:
        assert recordings_directory is not None
        print(
            f"Control: {algorithm}, recordings use best final seed {selected_seed}",
            flush=True,
        )
        recording_seeds, frame_duration, terminal_duration = recording_settings(
            environment
        )
        for checkpoint in recording_checkpoints:
            record_evaluation(
                recordings_directory / f"{algorithm}_checkpoint_{checkpoint:03d}.gif",
                environment=environment,
                encode_observation=encode_observation,
                action_values=recording_values[selected_seed][checkpoint],
                seeds=recording_seeds,
                frame_duration_ms=frame_duration,
                terminal_duration_ms=terminal_duration,
            )
            print(f"  recorded checkpoint {checkpoint}%", flush=True)
    return rows, selected_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default=KNOWN_ENVIRONMENTS[0])
    parser.add_argument(
        "--preset",
        choices=("quick", "tuning", "standard"),
        default="standard",
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        required=True,
        help="single prediction or control variant evaluated by this run",
    )
    parser.add_argument("--prediction-episodes", type=int)
    parser.add_argument("--training-episodes", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seeds", type=int)
    parser.add_argument(
        "--seed-base",
        type=int,
        help="first paired seed (default: 0 for quick; randomized for standard)",
    )
    args = parser.parse_args()
    defaults = experiment_defaults(args.environment, args.preset)
    prediction_episodes = (
        args.prediction_episodes
        if args.prediction_episodes is not None
        else defaults["prediction_episodes"]
    )
    training_episodes = (
        args.training_episodes
        if args.training_episodes is not None
        else defaults["training_episodes"]
    )
    evaluation_episodes = (
        args.evaluation_episodes
        if args.evaluation_episodes is not None
        else defaults["evaluation_episodes"]
    )
    seeds = args.seeds if args.seeds is not None else defaults["seeds"]
    if (
        min(
            prediction_episodes,
            training_episodes,
            evaluation_episodes,
            seeds,
        )
        < 1
    ):
        parser.error("episode counts and seeds must be positive")
    try:
        seed_values = resolve_seed_values(seeds, args.preset, args.seed_base)
    except ValueError as error:
        parser.error(str(error))
    if not 0 < args.epsilon <= 1:
        parser.error("epsilon must be greater than 0 and at most 1")

    try:
        number_of_states, number_of_actions, encoder, renderable = inspect_environment(
            args.environment
        )
    except (gym.error.Error, ValueError) as error:
        parser.error(str(error))

    output = (
        None
        if args.preset == "quick"
        else create_run_directory(
            "monte_carlo",
            args.environment,
            args.algorithm,
            args.preset,
        )
    )
    is_prediction = args.algorithm.endswith("_prediction")
    recording_checkpoints = (
        RECORDING_CHECKPOINTS
        if args.preset == "standard" and renderable and not is_prediction
        else ()
    )
    recordings_directory = output / "recordings" if output is not None else None
    if output is not None:
        (output / "figures").mkdir()
        if recording_checkpoints:
            assert recordings_directory is not None
            recordings_directory.mkdir()
    metrics_path = output / "metrics.csv" if output is not None else None
    recording_seeds, frame_duration, terminal_duration = recording_settings(
        args.environment
    )
    checkpoints = evaluation_checkpoints(args.preset)
    metadata = {
        "status": "running",
        "environment": args.environment,
        "preset": args.preset,
        "algorithm": args.algorithm,
        "prediction_episodes": prediction_episodes,
        "training_episodes": training_episodes,
        "evaluation_episodes": evaluation_episodes,
        "epsilon": args.epsilon,
        "seeds": seeds,
        "seed_values": list(seed_values),
        "number_of_states": number_of_states,
        "number_of_actions": number_of_actions,
        "prediction_policy": (
            "stick on 20 or 21, otherwise hit"
            if args.environment == "Blackjack-v1"
            else (
                "fixed safe path"
                if args.environment == "CliffWalking-v1"
                else "uniform random"
            )
        ),
        "evaluation_checkpoints": list(checkpoints),
        "recording_checkpoints": list(recording_checkpoints),
        "recording_environment_seeds": list(recording_seeds) if renderable else [],
        "recording_episodes": len(recording_seeds) if renderable else 0,
        "recording_greedy_tie_break_seed": 0,
        "recording_frame_duration_ms": frame_duration,
        "terminal_frame_duration_ms": terminal_duration,
        "evaluation_policy": "frozen greedy",
        "success_definition": success_definition(args.environment),
        "variability": "standard deviation across seed means",
        "figure_variability": "individual seed curves plus across-seed mean",
    }
    if output is not None:
        write_metadata(output / "metadata.json", metadata)

    if is_prediction:
        print(f"Prediction: {args.algorithm}", flush=True)
        rows = []
        prediction_values: dict[int, np.ndarray] = {}
        for seed in seed_values:
            seed_rows, values = run_prediction(
                args.algorithm,
                args.environment,
                number_of_states,
                number_of_actions,
                encoder,
                prediction_episodes,
                seed,
            )
            rows.extend(seed_rows)
            prediction_values[seed] = values
        if output is None:
            print(
                "Quick compatibility run complete; no artifacts were written.",
                flush=True,
            )
            return
        if args.preset == "standard":
            selected_seed = max(
                seed_values,
                key=lambda seed: sum(
                    int(row["visits"]) for row in rows if int(row["seed"]) == seed
                ),
            )
            np.savez_compressed(
                output / "best_model.npz",
                algorithm=np.asarray(args.algorithm),
                seed=np.asarray(selected_seed),
                V=prediction_values[selected_seed],
            )
            metadata["model_seed"] = selected_seed
            metadata["model_selection"] = "largest total prediction visit count"
            metadata["model_file"] = "best_model.npz"
        else:
            metadata["model_seed"] = None
            metadata["model_file"] = None
        metadata["selected_seed"] = selected_seed if args.preset == "standard" else None
        write_prediction_report(output, rows, metadata)
        metadata["status"] = "complete"
        write_metadata(output / "metadata.json", metadata)
        print(f"Complete: {output}", flush=True)
        return

    rows, selected_seed = run_control(
        args.algorithm,
        args.environment,
        number_of_states,
        number_of_actions,
        encoder,
        renderable,
        training_episodes,
        evaluation_episodes,
        args.epsilon,
        seed_values,
        recordings_directory,
        recording_checkpoints,
        checkpoints,
        metrics_path,
        [],
        output / "best_model.npz"
        if output is not None and args.preset == "standard"
        else None,
    )
    if output is None:
        print(
            "Quick compatibility run complete; no artifacts were written.",
            flush=True,
        )
        return
    metadata["recording_seed_selection"] = (
        "highest final mean evaluation return; success rate breaks ties"
        if recording_checkpoints
        else None
    )
    metadata["selected_seed"] = selected_seed if recording_checkpoints else None
    metadata["model_file"] = "best_model.npz" if args.preset == "standard" else None
    metadata["model_seed"] = selected_seed if args.preset == "standard" else None
    write_control_report(output, rows, metadata)
    metadata["status"] = "complete"
    write_metadata(output / "metadata.json", metadata)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
