"""Run tabular TD prediction and control on Gymnasium environments."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
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
from experiments.temporal_difference.environments import (
    KNOWN_ENVIRONMENTS,
    action_name,
    environment_configuration,
    episode_succeeded,
    experiment_defaults,
    fixed_prediction_action,
    make_environment,
    success_definition,
)
from experiments.temporal_difference.report import (
    CSV_FIELDS,
    write_control_report,
    write_prediction_report,
)
from rl_lib.algorithms.temporal_difference import (
    SARSA,
    QLearning,
    TDPrediction,
)
from rl_lib.trajectories import EpisodeStep

ALGORITHMS = ("td_prediction", "sarsa", "q_learning")


def greedy_action(Q: np.ndarray, rng: np.random.Generator, state: int) -> int:
    values = Q[state]
    return int(rng.choice(np.flatnonzero(values == np.max(values))))


def run_prediction(
    environment: str,
    configuration: dict[str, object],
    number_of_states: int,
    number_of_actions: int,
    episodes: int,
    learning_rate: float,
    discount: float,
    rollout_steps: int,
    seed: int,
    encode_observation: Callable[[Any], int] = int,
) -> list[dict[str, object]]:
    estimator = TDPrediction(
        number_of_states,
        learning_rate=learning_rate,
        discount=discount,
    )
    visits = np.zeros(number_of_states, dtype=int)
    rng = np.random.default_rng(seed)
    env = make_environment(environment, configuration)
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed * episodes + episode)
        state = encode_observation(observation)
        rollout: list[EpisodeStep[int]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = fixed_prediction_action(
                environment,
                rng,
                state,
                number_of_actions,
            )
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_observation(next_observation)
            rollout.append(EpisodeStep(state, action, float(reward)))
            visits[state] += 1
            state = next_state
            if len(rollout) == rollout_steps or terminated or truncated:
                estimator.update(tuple(rollout), state, terminated=terminated)
                rollout.clear()
    env.close()

    return [
        {
            "phase": "prediction",
            "algorithm": "td_prediction",
            "seed": seed,
            "checkpoint": 100,
            "state": state,
            "V": float(estimator.V[state]),
            "visits": int(visits[state]),
            "mean_return": "",
            "success_rate": "",
            "mean_episode_length": "",
            "truncation_rate": "",
            "mean_cliff_falls": "",
        }
        for state in range(number_of_states)
    ]


def train_episode(
    env: gym.Env,
    agent: SARSA | QLearning,
    algorithm: str,
    seed: int,
    rollout_steps: int,
    encode_observation: Callable[[Any], int] = int,
) -> None:
    observation, _ = env.reset(seed=seed)
    state = encode_observation(observation)
    action = agent.select_action(state)
    rollout: list[EpisodeStep[int]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = encode_observation(next_observation)
        rollout.append(EpisodeStep(state, action, float(reward)))
        if algorithm == "sarsa":
            next_action = None if terminated else agent.select_action(next_state)
            if len(rollout) == rollout_steps or terminated or truncated:
                agent.update(
                    tuple(rollout),
                    next_state,
                    next_action,
                    terminated=terminated,
                )
                rollout.clear()
        elif len(rollout) == rollout_steps or terminated or truncated:
            agent.update(tuple(rollout), next_state, terminated=terminated)
            rollout.clear()
            next_action = None if terminated else agent.select_action(next_state)
        else:
            next_action = agent.select_action(next_state)

        if not (terminated or truncated):
            state = next_state
            action = next_action


def evaluate(
    env: gym.Env,
    environment: str,
    Q: np.ndarray,
    episodes: int,
    seed: int,
    encode_observation: Callable[[Any], int],
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    successes: list[float] = []
    lengths: list[float] = []
    truncations: list[float] = []
    cliff_falls: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        state = encode_observation(observation)
        terminated = truncated = False
        total_reward = 0.0
        length = 0
        falls = 0
        while not (terminated or truncated):
            action = greedy_action(Q, rng, state)
            observation, reward, terminated, truncated, _ = env.step(action)
            state = encode_observation(observation)
            total_reward += float(reward)
            length += 1
            falls += environment == "CliffWalking-v1" and reward == -100
        returns.append(total_reward)
        success = episode_succeeded(environment, terminated, float(reward))
        successes.append(float(success))
        lengths.append(float(length))
        truncations.append(float(truncated))
        cliff_falls.append(float(falls))
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
        "truncation_rate": float(np.mean(truncations)),
        "mean_cliff_falls": float(np.mean(cliff_falls)),
    }


def record_evaluation(
    path: Path,
    environment: str,
    configuration: dict[str, object],
    Q: np.ndarray,
    seed: int,
    encode_observation: Callable[[Any], int],
) -> None:
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_environment(environment, configuration, render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    state = encode_observation(observation)
    frames = [annotated_frame(env.render(), "Evaluation: start")]
    durations = [150]
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = greedy_action(Q, rng, state)
        observation, reward, terminated, truncated, _ = env.step(action)
        state = encode_observation(observation)
        step += 1
        frames.append(
            annotated_frame(
                env.render(),
                f"Step {step}: {action_name(environment, action)}, "
                f"reward {float(reward):g}",
            )
        )
        durations.append(1_000 if terminated or truncated else 150)
    env.close()
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


def run_control(
    environment: str,
    algorithm: str,
    configurations: list[dict[str, object]],
    seed_values: tuple[int, ...],
    number_of_states: int,
    number_of_actions: int,
    encode_observation: Callable[[Any], int],
    training_episodes: int,
    evaluation_episodes: int,
    learning_rate: float,
    discount: float,
    epsilon: float,
    rollout_steps: int,
    recordings_directory: Path | None,
    metrics_path: Path | None,
    initial_rows: list[dict[str, object]],
    recording_checkpoints: tuple[int, ...],
    checkpoints: tuple[int, ...],
    model_path: Path | None,
) -> tuple[list[dict[str, object]], int]:
    rows = initial_rows.copy()
    agent_class = SARSA if algorithm == "sarsa" else QLearning
    recording_values: dict[int, dict[int, np.ndarray]] = {}
    for trial_index, (seed, configuration) in enumerate(
        zip(seed_values, configurations, strict=True)
    ):
        print(
            f"{algorithm}: seed trial {trial_index + 1}/{len(seed_values)} "
            f"(seed={seed})",
            flush=True,
        )
        env = make_environment(environment, configuration)
        agent = agent_class(
            number_of_states,
            number_of_actions,
            learning_rate=learning_rate,
            discount=discount,
            epsilon=epsilon,
            seed=seed,
        )
        completed_episodes = 0
        for checkpoint in checkpoints:
            target_episodes = checkpoint_episode_target(training_episodes, checkpoint)
            print(
                f"  checkpoint {checkpoint}%: train to {target_episodes}, "
                f"then evaluate {evaluation_episodes} episodes",
                flush=True,
            )
            while completed_episodes < target_episodes:
                train_episode(
                    env,
                    agent,
                    algorithm,
                    seed=seed * training_episodes + completed_episodes,
                    rollout_steps=rollout_steps,
                    encode_observation=encode_observation,
                )
                completed_episodes += 1

            measurements = evaluate(
                env,
                environment,
                agent.Q,
                evaluation_episodes,
                seed=1_000_000 + checkpoint * 10_000 + seed * evaluation_episodes,
                encode_observation=encode_observation,
            )
            rows.append(
                {
                    "phase": "control",
                    "algorithm": algorithm,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "state": "",
                    "V": "",
                    "visits": "",
                    **measurements,
                }
            )
            if metrics_path is not None:
                write_csv(metrics_path, CSV_FIELDS, rows)
            if checkpoint in recording_checkpoints or (
                model_path is not None and checkpoint == 100
            ):
                recording_values.setdefault(seed, {})[checkpoint] = agent.Q.copy()
        env.close()

    final_rows = [
        row for row in rows if row["phase"] == "control" and row["checkpoint"] == 100
    ]
    selected_row = max(
        final_rows,
        key=lambda row: (float(row["mean_return"]), float(row["success_rate"])),
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
    if recording_checkpoints:
        assert recordings_directory is not None
        print(
            f"{algorithm}: recordings use best final seed {selected_seed}",
            flush=True,
        )
        for checkpoint in recording_checkpoints:
            record_evaluation(
                recordings_directory / f"{algorithm}_checkpoint_{checkpoint:03d}.gif",
                environment,
                configurations[seed_values.index(selected_seed)],
                recording_values[selected_seed][checkpoint],
                seed=2_000_000 + checkpoint,
                encode_observation=encode_observation,
            )
            print(f"  recorded checkpoint {checkpoint}%", flush=True)
    return rows, selected_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        default=KNOWN_ENVIRONMENTS[0],
    )
    parser.add_argument(
        "--preset",
        choices=("quick", "tuning", "standard"),
        default="standard",
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        required=True,
        help="single algorithm evaluated by this run",
    )
    parser.add_argument("--prediction-episodes", type=int)
    parser.add_argument("--training-episodes", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument(
        "--seed-base",
        type=int,
        help="first paired seed (default: 0 for quick; randomized for standard)",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--rollout-steps",
        "--rollout",
        "--n-steps",
        "--n-step",
        dest="rollout_steps",
        type=int,
        default=1,
        help="maximum transitions per TD rollout (default: 1)",
    )
    parser.add_argument("--map-size", type=int, default=4)
    parser.add_argument("--safe-probability", type=float, default=0.8)
    parser.add_argument("--non-slippery", action="store_true")
    parser.add_argument("--max-episode-steps", type=int)
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
    if min(prediction_episodes, training_episodes, evaluation_episodes, seeds) < 1:
        parser.error("episode counts and seeds must be positive")
    try:
        seed_values = resolve_seed_values(seeds, args.preset, args.seed_base)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1:
        parser.error("learning rate must be finite and in (0, 1]")
    if not np.isfinite(args.discount) or not 0 <= args.discount <= 1:
        parser.error("discount must be finite and in [0, 1]")
    if not np.isfinite(args.epsilon) or not 0 <= args.epsilon <= 1:
        parser.error("epsilon must be finite and in [0, 1]")
    if args.rollout_steps < 1:
        parser.error("rollout steps must be positive")
    if args.map_size < 2:
        parser.error("map size must be at least 2")
    if not 0 < args.safe_probability <= 1:
        parser.error("safe probability must be in (0, 1]")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        parser.error("max episode steps must be positive")

    configurations, states, actions, max_steps, encoder = environment_configuration(
        args.environment,
        seed_values,
        args.map_size,
        args.safe_probability,
        not args.non_slippery,
        args.max_episode_steps,
    )
    output = (
        None
        if args.preset == "quick"
        else create_run_directory(
            "temporal_difference",
            args.environment,
            args.algorithm,
            args.preset,
        )
    )
    recordings = output / "recordings" if output is not None else None
    recording_checkpoints = (
        RECORDING_CHECKPOINTS
        if args.preset == "standard" and args.algorithm != "td_prediction"
        else ()
    )
    checkpoints = evaluation_checkpoints(args.preset)
    if output is not None:
        (output / "figures").mkdir()
        if recording_checkpoints:
            assert recordings is not None
            recordings.mkdir()
    metadata: dict[str, object] = {
        "status": "running",
        "environment": args.environment,
        "preset": args.preset,
        "algorithm": args.algorithm,
        "prediction_episodes": prediction_episodes,
        "training_episodes": training_episodes,
        "evaluation_episodes": evaluation_episodes,
        "seeds": seeds,
        "seed_values": list(seed_values),
        "number_of_states": states,
        "number_of_actions": actions,
        "learning_rate": args.learning_rate,
        "discount": args.discount,
        "epsilon": args.epsilon,
        "rollout_steps": args.rollout_steps,
        "max_episode_steps": max_steps,
        "evaluation_checkpoints": list(checkpoints),
        "recording_checkpoints": list(recording_checkpoints),
        "evaluation_policy": "frozen greedy",
        "prediction_policy": (
            "stick on 20 or 21, otherwise hit"
            if args.environment == "Blackjack-v1"
            else (
                "fixed safe path"
                if args.environment == "CliffWalking-v1"
                else "uniform random"
            )
        ),
        "success_definition": success_definition(args.environment),
        "truncation_target": "bootstrap then stop interaction",
        "variability": "standard deviation across seed evaluation means",
        "smoothing_window": 1,
        "frozen_lake": (
            {
                "map_size": args.map_size,
                "safe_probability": args.safe_probability,
                "is_slippery": not args.non_slippery,
                "maps": [configuration["desc"] for configuration in configurations],
            }
            if args.environment == "FrozenLake-v1"
            else None
        ),
    }
    if output is not None:
        write_metadata(output / "metadata.json", metadata)

    if args.algorithm == "td_prediction":
        prediction_rows = []
        for seed, configuration in zip(seed_values, configurations, strict=True):
            prediction_rows.extend(
                run_prediction(
                    args.environment,
                    configuration,
                    states,
                    actions,
                    prediction_episodes,
                    args.learning_rate,
                    args.discount,
                    args.rollout_steps,
                    seed,
                    encoder,
                )
            )
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
                    int(row["visits"])
                    for row in prediction_rows
                    if int(row["seed"]) == seed
                ),
            )
            selected = [
                row for row in prediction_rows if int(row["seed"]) == selected_seed
            ]
            np.savez_compressed(
                output / "best_model.npz",
                algorithm=np.asarray(args.algorithm),
                seed=np.asarray(selected_seed),
                V=np.asarray([row["V"] for row in selected], dtype=float),
            )
            metadata["model_seed"] = selected_seed
            metadata["model_selection"] = "largest total prediction visit count"
            metadata["model_file"] = "best_model.npz"
        else:
            metadata["model_seed"] = None
            metadata["model_file"] = None
        metadata["selected_seed"] = selected_seed if args.preset == "standard" else None
        write_prediction_report(output, prediction_rows, metadata)
        metadata["status"] = "complete"
        write_metadata(output / "metadata.json", metadata)
        print(f"Complete: {output}", flush=True)
        return

    prediction_rows: list[dict[str, object]] = []
    rows, selected_seed = run_control(
        args.environment,
        args.algorithm,
        configurations,
        seed_values,
        states,
        actions,
        encoder,
        training_episodes,
        evaluation_episodes,
        args.learning_rate,
        args.discount,
        args.epsilon,
        args.rollout_steps,
        recordings,
        output / "metrics.csv" if output is not None else None,
        prediction_rows,
        recording_checkpoints,
        checkpoints,
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
