"""Run Monte Carlo prediction and control on tabular Gymnasium environments."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    EVALUATION_CHECKPOINTS,
    RECORDING_CHECKPOINTS,
    annotated_frame,
    checkpoint_episode_target,
    create_run_directory,
    recording_title,
    write_csv,
    write_metadata,
)
from experiments.summary import SummaryMedia, SummaryTable, write_summary
from rl_lib.algorithms.monte_carlo import (
    EveryVisitMonteCarloControl,
    EveryVisitMonteCarloPrediction,
    FirstVisitMonteCarloControl,
    FirstVisitMonteCarloPrediction,
)
from rl_lib.data.episode import Episode, EpisodeStep

BLACKJACK_STATES = 32 * 11 * 2
BLACKJACK_ACTIONS = 2
RECORDING_SEEDS = (4, 5, 6, 7, 8)


def experiment_defaults(environment: str, preset: str) -> dict[str, int]:
    """Return episode budgets suited to the environment's typical horizon."""
    if preset == "quick":
        if environment == "Blackjack-v1":
            return {
                "prediction_episodes": 500,
                "training_episodes": 500,
                "evaluation_episodes": 100,
                "seeds": 1,
            }
        return {
            "prediction_episodes": 100,
            "training_episodes": 250,
            "evaluation_episodes": 25,
            "seeds": 1,
        }
    if environment == "Blackjack-v1":
        return {
            "prediction_episodes": 50_000,
            "training_episodes": 50_000,
            "evaluation_episodes": 5_000,
            "seeds": 3,
        }
    if environment == "Taxi-v4":
        return {
            "prediction_episodes": 2_000,
            "training_episodes": 10_000,
            "evaluation_episodes": 100,
            "seeds": 3,
        }
    return {
        "prediction_episodes": 2_000,
        "training_episodes": 5_000,
        "evaluation_episodes": 250,
        "seeds": 3,
    }


def recording_settings(environment: str) -> tuple[tuple[int, ...], int, int]:
    if environment == "Blackjack-v1":
        return RECORDING_SEEDS, 750, 1_500
    return RECORDING_SEEDS[:1], 150, 1_000


def action_name(environment: str, action: int) -> str:
    if environment == "Blackjack-v1":
        return ("stick", "hit")[action]
    if environment == "Taxi-v4":
        return ("south", "north", "east", "west", "pickup", "drop-off")[action]
    return str(action)


def encode_blackjack_state(observation: tuple[int, int, int]) -> int:
    player, dealer, usable_ace = observation
    return (player * 11 + dealer) * 2 + int(usable_ace)


def encode_discrete_state(observation: Any) -> int:
    return int(observation)


def inspect_environment(
    environment: str,
) -> tuple[int, int, Callable[[Any], int], bool]:
    """Return the tabular dimensions, encoder, and rendering support."""
    env = gym.make(environment)
    renderable = "rgb_array" in env.metadata.get("render_modes", [])
    if environment == "Blackjack-v1":
        result = (
            BLACKJACK_STATES,
            BLACKJACK_ACTIONS,
            encode_blackjack_state,
            renderable,
        )
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        if not isinstance(observation_space, gym.spaces.Discrete) or not isinstance(
            action_space, gym.spaces.Discrete
        ):
            env.close()
            raise ValueError(
                f"{environment} must have Discrete observation and action spaces"
            )
        if observation_space.start != 0 or action_space.start != 0:
            env.close()
            raise ValueError(f"{environment} must use zero-based discrete spaces")
        result = (
            int(observation_space.n),
            int(action_space.n),
            encode_discrete_state,
            renderable,
        )
    env.close()
    return result


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


def fixed_blackjack_action(state: int) -> int:
    player = state // 22
    return 0 if player >= 20 else 1


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
    environment: str,
    number_of_states: int,
    number_of_actions: int,
    encode_observation: Callable[[Any], int],
    episodes: int,
    seed: int,
) -> list[dict[str, object]]:
    first = FirstVisitMonteCarloPrediction(number_of_states)
    every = EveryVisitMonteCarloPrediction(number_of_states)
    rng = np.random.default_rng(seed)
    select_action = (
        fixed_blackjack_action
        if environment == "Blackjack-v1"
        else lambda _state: int(rng.integers(number_of_actions))
    )
    env = gym.make(environment)
    for episode in range(episodes):
        trajectory = generate_episode(
            env,
            select_action,
            seed=seed + episode,
            encode_observation=encode_observation,
        )
        first.update(trajectory)
        every.update(trajectory)
    env.close()

    rows = []
    for state in prediction_states(environment, number_of_states):
        for name, estimator in (
            ("first_visit", first),
            ("every_visit", every),
        ):
            rows.append(
                {
                    "phase": "prediction",
                    "algorithm": name,
                    "seed": 0,
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
            )
    return rows


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
    env = gym.make(environment, render_mode="rgb_array")
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
    environment: str,
    number_of_states: int,
    number_of_actions: int,
    encode_observation: Callable[[Any], int],
    renderable: bool,
    training_episodes: int,
    evaluation_episodes: int,
    epsilon: float,
    seeds: int,
    recordings_directory: Path,
    recording_checkpoints: tuple[int, ...],
    metrics_path: Path,
    initial_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows = initial_rows.copy()
    selected_recording_seeds: dict[str, int] = {}
    classes = (
        ("first_visit", FirstVisitMonteCarloControl),
        ("every_visit", EveryVisitMonteCarloControl),
    )
    for name, control_class in classes:
        recording_values: dict[int, dict[int, np.ndarray]] = {}
        algorithm_rows: dict[int, dict[str, object]] = {}
        checkpoint_metrics: dict[str, dict[int, list[float]]] = {
            metric: {checkpoint: [] for checkpoint in EVALUATION_CHECKPOINTS}
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
        for seed in range(seeds):
            print(f"Control: {name}, seed {seed + 1}/{seeds}", flush=True)
            env = gym.make(environment)
            agent = control_class(
                number_of_states,
                number_of_actions,
                epsilon=epsilon,
                seed=seed,
            )
            completed_episodes = 0
            for checkpoint in EVALUATION_CHECKPOINTS:
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
                seed_returns = []
                seed_successes = []
                seed_lengths = []
                seed_truncations = []
                seed_illegal_actions = []
                seed_draws = []
                seed_losses = []
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
                    seed_returns.append(episode_return)
                    seed_lengths.append(len(trajectory.steps))
                    seed_truncations.append(trajectory.truncated)
                    if environment in ("Blackjack-v1", "Taxi-v4"):
                        seed_successes.append(trajectory.steps[-1].reward > 0)
                    if environment == "Taxi-v4":
                        seed_illegal_actions.append(
                            sum(step.reward == -10 for step in trajectory.steps)
                        )
                    if environment == "Blackjack-v1":
                        seed_draws.append(episode_return == 0)
                        seed_losses.append(episode_return < 0)

                samples = {
                    "return": seed_returns,
                    "success": seed_successes,
                    "length": seed_lengths,
                    "truncation": seed_truncations,
                    "illegal_actions": seed_illegal_actions,
                    "draw": seed_draws,
                    "loss": seed_losses,
                }
                for metric, values in samples.items():
                    if values:
                        checkpoint_metrics[metric][checkpoint].append(
                            float(np.mean(values))
                        )
                rows.append(
                    _control_seed_row(
                        name,
                        seed,
                        checkpoint,
                        samples,
                        evaluation_episodes,
                    )
                )
                algorithm_rows[checkpoint] = _control_metric_row(
                    name,
                    checkpoint,
                    checkpoint_metrics,
                    evaluation_episodes,
                    seed + 1,
                )
                completed_rows = rows + [
                    algorithm_rows[item]
                    for item in EVALUATION_CHECKPOINTS
                    if item in algorithm_rows
                ]
                write_csv(metrics_path, tuple(rows[0]), completed_rows)

                if renderable and checkpoint in recording_checkpoints:
                    recording_values.setdefault(seed, {})[checkpoint] = agent.Q.copy()
            env.close()
        rows.extend(algorithm_rows[checkpoint] for checkpoint in EVALUATION_CHECKPOINTS)
        write_csv(metrics_path, tuple(rows[0]), rows)

        final_seed_rows = [
            row
            for row in rows
            if row["phase"] == "control_seed"
            and row["algorithm"] == name
            and row["checkpoint"] == 100
        ]
        selected_row = max(
            final_seed_rows,
            key=lambda row: (
                float(row["mean_return"]),
                float(row["success_rate"]) if row["success_rate"] != "" else 0.0,
            ),
        )
        selected_seed = int(selected_row["seed"])
        selected_recording_seeds[name] = selected_seed
        if renderable and recording_checkpoints:
            print(
                f"Control: {name}, recordings use best final seed {selected_seed}",
                flush=True,
            )
            recording_seeds, frame_duration, terminal_duration = recording_settings(
                environment
            )
            for checkpoint in recording_checkpoints:
                record_evaluation(
                    recordings_directory / f"{name}_checkpoint_{checkpoint:03d}.gif",
                    environment=environment,
                    encode_observation=encode_observation,
                    action_values=recording_values[selected_seed][checkpoint],
                    seeds=recording_seeds,
                    frame_duration_ms=frame_duration,
                    terminal_duration_ms=terminal_duration,
                )
                print(f"  recorded checkpoint {checkpoint}%", flush=True)
    return rows, selected_recording_seeds


def _write_prediction_figure(
    output: Path,
    rows: list[dict[str, object]],
    environment: str,
) -> None:
    prediction_rows = [row for row in rows if row["phase"] == "prediction"]
    if environment == "Blackjack-v1":
        value_limit = max(abs(float(row["value"])) for row in prediction_rows)
        value_limit = max(value_limit, 0.01)
        figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        image = None
        for column, algorithm in enumerate(("first_visit", "every_visit")):
            for row_index, usable_ace in enumerate((0, 1)):
                values = np.empty((10, 10), dtype=float)
                for row in prediction_rows:
                    state = int(row["state"])
                    player = state // 22
                    dealer = (state % 22) // 2
                    ace = state % 2
                    if row["algorithm"] == algorithm and ace == usable_ace:
                        values[player - 12, dealer - 1] = float(row["value"])
                axis = axes[row_index, column]
                image = axis.imshow(
                    values,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-value_limit,
                    vmax=value_limit,
                    aspect="auto",
                )
                axis.set_title(
                    f"{algorithm.replace('_', ' ').title()}, "
                    f"{'usable ace' if usable_ace else 'no usable ace'}"
                )
                axis.set_xticks(range(10), range(1, 11))
                axis.set_yticks(range(10), range(12, 22))
                axis.set_xlabel("Dealer showing")
                axis.set_ylabel("Player total")
        figure.colorbar(image, ax=axes, label="Estimated state value", shrink=0.85)
    else:
        figure, axes = plt.subplots(2, 1, figsize=(11, 7))
        for algorithm in ("first_visit", "every_visit"):
            selected = [row for row in prediction_rows if row["algorithm"] == algorithm]
            values = np.sort(
                [float(row["value"]) for row in selected if int(row["visits"]) > 0]
            )
            visits = np.sort(
                [int(row["visits"]) for row in selected if int(row["visits"]) > 0]
            )[::-1]
            axes[0].plot(
                np.linspace(0, 100, len(values)),
                values,
                linewidth=1,
                label=algorithm,
            )
            axes[1].plot(
                np.arange(1, len(visits) + 1),
                visits,
                linewidth=1,
                label=algorithm,
            )
        axes[0].set_ylabel("Estimated value")
        axes[0].set_xlabel("Visited-state percentile")
        axes[1].set_ylabel("Visits")
        axes[1].set_xlabel("Visited states, sorted by count")
        axes[1].set_yscale("log")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
    figure.suptitle("Fixed-policy Monte Carlo prediction")
    figure.savefig(output / "figures" / "prediction_values.png", dpi=150)
    plt.close(figure)


def _write_figures(
    output: Path,
    rows: list[dict[str, object]],
    seeds: int,
    environment: str,
) -> None:
    figures = output / "figures"
    figures.mkdir()
    _write_prediction_figure(output, rows, environment)

    control_rows = [row for row in rows if row["phase"] == "control"]
    if environment == "Blackjack-v1":
        metrics = (
            ("mean_return", "return_std", "Mean evaluation return"),
            ("success_rate", "success_rate_std", "Win rate"),
        )
    elif environment == "Taxi-v4":
        metrics = (
            ("mean_return", "return_std", "Mean evaluation return"),
            ("success_rate", "success_rate_std", "Success rate"),
            (
                "mean_episode_length",
                "episode_length_std",
                "Mean episode length",
            ),
            ("truncation_rate", "truncation_rate_std", "Truncation rate"),
        )
    else:
        metrics = (
            ("mean_return", "return_std", "Mean evaluation return"),
            (
                "mean_episode_length",
                "episode_length_std",
                "Mean episode length",
            ),
            ("truncation_rate", "truncation_rate_std", "Truncation rate"),
        )
    columns = 2 if len(metrics) in (2, 4) else 3
    rows_count = (len(metrics) + columns - 1) // columns
    figure, axes_grid = plt.subplots(
        rows_count,
        columns,
        figsize=(5.5 * columns, 4.2 * rows_count),
        squeeze=False,
    )
    axes = axes_grid.ravel()
    for algorithm in ("first_visit", "every_visit"):
        selected = [row for row in control_rows if row["algorithm"] == algorithm]
        checkpoints = np.asarray([int(row["checkpoint"]) for row in selected])
        for axis, (mean_field, std_field, label) in zip(axes, metrics, strict=False):
            means = np.asarray([float(row[mean_field]) for row in selected])
            errors = np.asarray([float(row[std_field]) for row in selected]) / np.sqrt(
                seeds
            )
            (line,) = axis.plot(checkpoints, means, marker="o", label=algorithm)
            axis.fill_between(
                checkpoints,
                means - 1.96 * errors,
                means + 1.96 * errors,
                color=line.get_color(),
                alpha=0.12,
            )
            axis.set_xlabel("Training completed (%)")
            axis.set_ylabel(label)
            axis.set_xticks(EVALUATION_CHECKPOINTS)
            axis.grid(alpha=0.25)
            axis.legend()
    for axis, (mean_field, _, _) in zip(axes, metrics, strict=False):
        if mean_field in ("success_rate", "truncation_rate"):
            axis.set_ylim(0, 1)
    for axis in axes[len(metrics) :]:
        axis.remove()
    figure.suptitle("Frozen greedy-policy evaluation")
    figure.tight_layout()
    figure.savefig(figures / "control_learning.png", dpi=150)
    plt.close(figure)


def _mean_and_std(row: dict[str, object], mean: str, std: str) -> str:
    return f"{float(row[mean]):.3f} +/- {float(row[std]):.3f}"


def _control_table(
    environment: str,
    rows: list[dict[str, object]],
) -> SummaryTable:
    if environment == "Blackjack-v1":
        headers = (
            "Algorithm",
            "Checkpoint",
            "Mean return",
            "Win rate",
            "Draw rate",
            "Loss rate",
        )
        values = tuple(
            (
                row["algorithm"],
                f"{row['checkpoint']}%",
                _mean_and_std(row, "mean_return", "return_std"),
                _mean_and_std(row, "success_rate", "success_rate_std"),
                f"{float(row['draw_rate']):.3f}",
                f"{float(row['loss_rate']):.3f}",
            )
            for row in rows
        )
    elif environment == "Taxi-v4":
        headers = (
            "Algorithm",
            "Checkpoint",
            "Mean return",
            "Success rate",
            "Episode length",
            "Truncation rate",
            "Illegal actions",
        )
        values = tuple(
            (
                row["algorithm"],
                f"{row['checkpoint']}%",
                _mean_and_std(row, "mean_return", "return_std"),
                _mean_and_std(row, "success_rate", "success_rate_std"),
                _mean_and_std(row, "mean_episode_length", "episode_length_std"),
                _mean_and_std(row, "truncation_rate", "truncation_rate_std"),
                _mean_and_std(row, "illegal_actions", "illegal_actions_std"),
            )
            for row in rows
        )
    else:
        headers = (
            "Algorithm",
            "Checkpoint",
            "Mean return",
            "Episode length",
            "Truncation rate",
        )
        values = tuple(
            (
                row["algorithm"],
                f"{row['checkpoint']}%",
                _mean_and_std(row, "mean_return", "return_std"),
                _mean_and_std(row, "mean_episode_length", "episode_length_std"),
                _mean_and_std(row, "truncation_rate", "truncation_rate_std"),
            )
            for row in rows
        )
    return SummaryTable("Control evaluation", headers, values)


def write_outputs(output: Path, rows: list[dict[str, object]], metadata: dict) -> None:
    write_csv(output / "metrics.csv", tuple(rows[0]), rows)
    write_metadata(output / "metadata.json", metadata)
    environment = str(metadata["environment"])
    _write_figures(output, rows, int(metadata["seeds"]), environment)
    prediction_rows = [row for row in rows if row["phase"] == "prediction"]
    control_rows = [row for row in rows if row["phase"] == "control"]
    seed_rows = [row for row in rows if row["phase"] == "control_seed"]
    prediction_summary = []
    for algorithm in ("first_visit", "every_visit"):
        selected = [row for row in prediction_rows if row["algorithm"] == algorithm]
        visited_values = [
            float(row["value"]) for row in selected if int(row["visits"]) > 0
        ]
        value_range = (
            f"{min(visited_values):.3f} to {max(visited_values):.3f}"
            if visited_values
            else "not visited"
        )
        prediction_summary.append(
            (
                algorithm,
                len(selected),
                len(visited_values),
                sum(int(row["visits"]) for row in selected),
                value_range,
            )
        )
    recordings = tuple(
        SummaryMedia(
            recording_title(path, metadata["recording_seed_by_algorithm"]),
            path.relative_to(output),
        )
        for path in sorted(
            (output / "recordings").glob("*.gif"),
            key=lambda item: (
                0 if item.stem.startswith("first_visit") else 1,
                item.name,
            ),
        )
    )
    write_summary(
        output / "summary.html",
        title=f"Monte Carlo on {environment}",
        metadata={
            "Environment": environment,
            "Preset": metadata["preset"],
            "Prediction policy": metadata["prediction_policy"],
            "Prediction episodes": metadata["prediction_episodes"],
            "Control training episodes": metadata["training_episodes"],
            "Evaluation": (
                f"{metadata['evaluation_episodes']} episodes per checkpoint and seed"
            ),
            "Seeds": metadata["seeds"],
            "Training epsilon": metadata["epsilon"],
            "Recorded seed by algorithm": metadata["recording_seed_by_algorithm"],
        },
        tables=(
            SummaryTable(
                "Prediction coverage",
                (
                    "Algorithm",
                    "Reported states",
                    "Visited states",
                    "Total visits",
                    "Value range",
                ),
                prediction_summary,
            ),
            _control_table(environment, control_rows),
            SummaryTable(
                "Final control evaluation by seed",
                (
                    "Algorithm",
                    "Seed",
                    "Mean return",
                    "Success rate",
                    "Episode length",
                    "Truncation rate",
                ),
                tuple(
                    (
                        row["algorithm"],
                        row["seed"],
                        f"{float(row['mean_return']):.3f}",
                        (
                            f"{float(row['success_rate']):.3f}"
                            if row["success_rate"] != ""
                            else "not defined"
                        ),
                        f"{float(row['mean_episode_length']):.2f}",
                        f"{float(row['truncation_rate']):.3f}",
                    )
                    for row in seed_rows
                    if row["checkpoint"] == 100
                ),
            ),
        ),
        figures=(
            SummaryMedia(
                "Control checkpoint evaluation", Path("figures/control_learning.png")
            ),
            SummaryMedia("Prediction values", Path("figures/prediction_values.png")),
        ),
        recordings=recordings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="Blackjack-v1")
    parser.add_argument("--preset", choices=("quick", "standard"), default="standard")
    parser.add_argument("--prediction-episodes", type=int)
    parser.add_argument("--training-episodes", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seeds", type=int)
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
    if not 0 < args.epsilon <= 1:
        parser.error("epsilon must be greater than 0 and at most 1")

    try:
        number_of_states, number_of_actions, encoder, renderable = inspect_environment(
            args.environment
        )
    except (gym.error.Error, ValueError) as error:
        parser.error(str(error))

    output = create_run_directory("monte_carlo", args.environment)
    recordings_directory = output / "recordings"
    recordings_directory.mkdir()
    metrics_path = output / "metrics.csv"
    recording_seeds, frame_duration, terminal_duration = recording_settings(
        args.environment
    )
    metadata = {
        "status": "running",
        "environment": args.environment,
        "preset": args.preset,
        "prediction_episodes": prediction_episodes,
        "training_episodes": training_episodes,
        "evaluation_episodes": evaluation_episodes,
        "epsilon": args.epsilon,
        "seeds": seeds,
        "number_of_states": number_of_states,
        "number_of_actions": number_of_actions,
        "prediction_policy": (
            "stick on 20 or 21, otherwise hit"
            if args.environment == "Blackjack-v1"
            else "uniform random"
        ),
        "evaluation_checkpoints": list(EVALUATION_CHECKPOINTS),
        "recording_checkpoints": (
            [100]
            if renderable and args.preset == "quick"
            else list(RECORDING_CHECKPOINTS)
            if renderable
            else []
        ),
        "recording_environment_seeds": list(recording_seeds) if renderable else [],
        "recording_episodes": len(recording_seeds) if renderable else 0,
        "recording_greedy_tie_break_seed": 0,
        "recording_frame_duration_ms": frame_duration,
        "terminal_frame_duration_ms": terminal_duration,
        "evaluation_policy": "frozen greedy",
        "success_definition": (
            "positive terminal reward"
            if args.environment in ("Blackjack-v1", "Taxi-v4")
            else None
        ),
        "variability": "standard deviation across seed means",
        "confidence_interval": "mean +/- 1.96 * standard error",
    }
    write_metadata(output / "metadata.json", metadata)
    print("Prediction: fixed policy", flush=True)
    rows = run_prediction(
        args.environment,
        number_of_states,
        number_of_actions,
        encoder,
        prediction_episodes,
        seed=0,
    )
    write_csv(metrics_path, tuple(rows[0]), rows)
    rows, selected_recording_seeds = run_control(
        args.environment,
        number_of_states,
        number_of_actions,
        encoder,
        renderable,
        training_episodes,
        evaluation_episodes,
        args.epsilon,
        seeds,
        recordings_directory,
        (100,) if args.preset == "quick" else RECORDING_CHECKPOINTS,
        metrics_path,
        rows,
    )
    metadata["recording_seed_selection"] = (
        "highest final mean evaluation return; success rate breaks ties"
    )
    metadata["recording_seed_by_algorithm"] = selected_recording_seeds
    write_outputs(output, rows, metadata)
    metadata["status"] = "complete"
    write_metadata(output / "metadata.json", metadata)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
