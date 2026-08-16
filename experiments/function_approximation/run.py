"""Run semi-gradient control on continuous-observation Gymnasium environments."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

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
from experiments.function_approximation.diagnostics import (
    EpisodeDiagnostics,
    EpisodeDiagnosticTracker,
)
from experiments.summary import SummaryMedia, SummaryTable, write_summary
from rl_lib.algorithms.function_approximation import (
    SemiGradientQLearning,
    SemiGradientSARSA,
)
from rl_lib.data import EpisodeStep
from rl_lib.models import ActionValueNetwork

ENVIRONMENTS = ("Acrobot-v1", "MountainCar-v0")
ALGORITHMS = ("sarsa", "q_learning")
DEFAULT_HIDDEN_SIZES = (64, 64)
CSV_FIELDS = (
    "algorithm",
    "seed",
    "checkpoint",
    "evaluation_kind",
    "policy_checkpoint",
    "evaluation_episode",
    "episode_return",
    "episode_length",
    "success",
    "terminated",
    "truncated",
    "progress",
)
DIAGNOSTIC_FIELDS = (
    "algorithm",
    "seed",
    "training_episode",
    "epsilon",
    "learning_rate",
    "episode_return",
    "episode_length",
    "terminated",
    "truncated",
    "mean_absolute_td_error",
    "maximum_absolute_td_error",
    "mean_absolute_q_value",
    "maximum_absolute_q_value",
    "mean_action_gap",
    "dominant_action_fraction",
    "action_counts",
)
Agent = SemiGradientSARSA | SemiGradientQLearning
ModelState = dict[str, torch.Tensor]


def linearly_decayed_epsilon(
    initial_epsilon: float,
    final_epsilon: float,
    episode_index: int,
    training_episodes: int,
) -> float:
    """Interpolate epsilon from the first through the final training episode."""
    if training_episodes < 1:
        raise ValueError("training_episodes must be positive")
    if not 0 <= episode_index < training_episodes:
        raise ValueError("episode_index must identify a training episode")

    if training_episodes == 1:
        return initial_epsilon
    progress = episode_index / (training_episodes - 1)
    return initial_epsilon + progress * (final_epsilon - initial_epsilon)


def algorithm_final_epsilon(
    algorithm: str,
    initial_epsilon: float,
    sarsa_final_epsilon: float,
) -> float:
    """Keep Q-learning exploratory while SARSA approaches its greedy policy."""
    if algorithm == "sarsa":
        return sarsa_final_epsilon
    if algorithm == "q_learning":
        return initial_epsilon
    raise ValueError("algorithm must be 'sarsa' or 'q_learning'")


def optimizer_learning_rate(agent: Agent) -> float:
    """Return the learning rate used by the agent's single parameter group."""
    if len(agent.optimizer.param_groups) != 1:
        raise ValueError("function-approximation optimizers need one parameter group")
    return float(agent.optimizer.param_groups[0]["lr"])


def copy_model_state(model: torch.nn.Module) -> ModelState:
    """Copy model parameters and buffers without retaining graph references."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def make_environment(
    environment: str,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment must be one of {ENVIRONMENTS}")

    env = gym.make(environment, render_mode=render_mode)
    if not isinstance(env.observation_space, gym.spaces.Box):
        env.close()
        raise ValueError("environment must have a continuous Box observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise ValueError("environment must have a discrete action space")
    if not (
        np.all(np.isfinite(env.observation_space.low))
        and np.all(np.isfinite(env.observation_space.high))
    ):
        env.close()
        raise ValueError("environment observation bounds must be finite")

    return gym.wrappers.RescaleObservation(
        env,
        np.float32(-1.0),
        np.float32(1.0),
    )


def make_agent(
    algorithm: str,
    env: gym.Env,
    *,
    learning_rate: float,
    discount: float,
    epsilon: float,
    optimizer_name: str,
    hidden_sizes: tuple[int, ...],
    seed: int,
) -> Agent:
    if algorithm not in ("sarsa", "q_learning"):
        raise ValueError("algorithm must be 'sarsa' or 'q_learning'")
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("environment must have a Box observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("environment must have a Discrete action space")

    observation_size = int(np.prod(env.observation_space.shape))
    number_of_actions = int(env.action_space.n)

    torch.manual_seed(seed)
    model = ActionValueNetwork(
        observation_size,
        number_of_actions,
        hidden_sizes=hidden_sizes,
    )
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    agent_class = SemiGradientSARSA if algorithm == "sarsa" else SemiGradientQLearning
    return agent_class(
        model,
        optimizer,
        discount=discount,
        epsilon=epsilon,
        seed=seed,
    )


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    diagnostics: EpisodeDiagnostics | None = None


@dataclass(frozen=True)
class EvaluationResult:
    episode_return: float
    episode_length: int
    success: bool
    terminated: bool
    truncated: bool
    progress: float


def train_episode(
    env: gym.Env,
    agent: Agent,
    *,
    seed: int,
    rollout_steps: int,
    collect_diagnostics: bool = False,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed)
    action = agent.select_action(observation)
    rollout: list[EpisodeStep[np.ndarray]] = []
    episode_return = 0.0
    episode_length = 0
    tracker = (
        EpisodeDiagnosticTracker(agent.model.number_of_actions)
        if collect_diagnostics
        else None
    )

    terminated = truncated = False
    while not (terminated or truncated):
        current_action = action
        pre_update_action_values = (
            model_action_values(agent, observation) if tracker is not None else None
        )
        next_observation, reward, terminated, truncated, _ = env.step(current_action)
        rollout.append(
            EpisodeStep(
                np.asarray(observation, dtype=np.float32).copy(),
                current_action,
                float(reward),
            )
        )
        episode_return += float(reward)
        episode_length += 1

        should_update = len(rollout) == rollout_steps or terminated or truncated
        td_errors: tuple[float, ...] = ()
        if isinstance(agent, SemiGradientSARSA):
            if terminated:
                next_action = None
            else:
                next_action = agent.select_action(next_observation)
            if should_update:
                td_errors = agent.update(
                    tuple(rollout),
                    next_observation,
                    next_action,
                    terminated=terminated,
                )
        elif should_update:
            td_errors = agent.update(
                tuple(rollout),
                next_observation,
                terminated=terminated,
            )

        if tracker is not None:
            assert pre_update_action_values is not None
            tracker.record_transition(
                reward=float(reward),
                action=current_action,
                action_values=pre_update_action_values,
            )
            tracker.record_td_errors(td_errors)

        if should_update:
            rollout.clear()

        if terminated or truncated:
            break

        observation = next_observation
        if isinstance(agent, SemiGradientSARSA):
            assert next_action is not None
            action = next_action
        else:
            action = agent.select_action(observation)

    return EpisodeResult(
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=terminated,
        truncated=truncated,
        diagnostics=tracker.finish() if tracker is not None else None,
    )


def model_action_values(agent: Agent, observation: np.ndarray) -> np.ndarray:
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        action_values = agent.model(observation_tensor)
    return action_values.detach().cpu().numpy()


def greedy_action(agent: Agent, observation: np.ndarray) -> int:
    return int(np.argmax(model_action_values(agent, observation)))


def progress_value(environment: str, observation: np.ndarray) -> float:
    if environment == "MountainCar-v0":
        scaled_position = float(observation[0])
        return -1.2 + 0.9 * (scaled_position + 1.0)

    cos_first, sin_first, cos_second, sin_second = map(float, observation[:4])
    cos_combined = cos_first * cos_second - sin_first * sin_second
    return -cos_first - cos_combined


def progress_configuration(environment: str) -> tuple[str, float, str]:
    if environment == "MountainCar-v0":
        return (
            "Maximum position",
            0.5,
            "true termination after reaching position 0.5",
        )
    return (
        "Maximum tip height",
        1.0,
        "true termination after tip height exceeds 1.0",
    )


def evaluate_episode(
    env: gym.Env,
    environment: str,
    agent: Agent,
    *,
    seed: int,
) -> EvaluationResult:
    was_training = agent.model.training
    agent.model.eval()
    try:
        observation, _ = env.reset(seed=seed)
        maximum_progress = progress_value(environment, observation)
        episode_return = 0.0
        episode_length = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = greedy_action(agent, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            maximum_progress = max(
                maximum_progress,
                progress_value(environment, observation),
            )
    finally:
        agent.model.train(was_training)

    return EvaluationResult(
        episode_return=episode_return,
        episode_length=episode_length,
        success=terminated,
        terminated=terminated,
        truncated=truncated,
        progress=maximum_progress,
    )


def evaluation_score(
    results: list[EvaluationResult],
) -> tuple[float, float, float]:
    """Rank validation policies by success, then return, then progress."""
    if not results:
        raise ValueError("at least one validation result is required")
    return (
        float(np.mean([result.success for result in results])),
        float(np.mean([result.episode_return for result in results])),
        float(np.mean([result.progress for result in results])),
    )


def evaluation_row(
    algorithm: str,
    seed: int,
    checkpoint: int,
    evaluation_kind: str,
    policy_checkpoint: int,
    evaluation_episode: int,
    result: EvaluationResult,
) -> dict[str, object]:
    """Convert one frozen-policy episode to the shared metrics schema."""
    return {
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint": checkpoint,
        "evaluation_kind": evaluation_kind,
        "policy_checkpoint": policy_checkpoint,
        "evaluation_episode": evaluation_episode,
        "episode_return": result.episode_return,
        "episode_length": result.episode_length,
        "success": int(result.success),
        "terminated": int(result.terminated),
        "truncated": int(result.truncated),
        "progress": result.progress,
    }


def rendered_frame(env: gym.Env, label: str) -> Image.Image:
    frame = env.render()
    if not isinstance(frame, np.ndarray):
        raise RuntimeError("environment did not return an RGB frame")
    return annotated_frame(frame, label)


def action_name(environment: str, action: int) -> str:
    if environment == "MountainCar-v0":
        return ("push left", "no push", "push right")[action]
    return ("negative torque", "no torque", "positive torque")[action]


def record_evaluation(
    path: Path,
    environment: str,
    agent: Agent,
    *,
    seed: int,
    frame_stride: int,
) -> None:
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_environment(environment, render_mode="rgb_array")
    was_training = agent.model.training
    agent.model.eval()
    frames: list[Image.Image] = []
    durations: list[int] = []
    try:
        observation, _ = env.reset(seed=seed)
        frames.append(rendered_frame(env, "Frozen greedy evaluation: start"))
        durations.append(100)
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            action = greedy_action(agent, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            step += 1
            if step % frame_stride == 0 or terminated or truncated:
                frames.append(
                    rendered_frame(
                        env,
                        f"Step {step}: {action_name(environment, action)}, "
                        f"reward {float(reward):g}",
                    )
                )
                durations.append(1_000 if terminated or truncated else 100)
    finally:
        agent.model.train(was_training)
        env.close()

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


def diagnostic_row(
    algorithm: str,
    seed: int,
    training_episode: int,
    epsilon: float,
    learning_rate: float,
    result: EpisodeResult,
) -> dict[str, object]:
    diagnostics = result.diagnostics
    if diagnostics is None:
        raise ValueError("episode result does not contain diagnostics")

    return {
        "algorithm": algorithm,
        "seed": seed,
        "training_episode": training_episode,
        "epsilon": epsilon,
        "learning_rate": learning_rate,
        "episode_return": diagnostics.episode_return,
        "episode_length": diagnostics.episode_length,
        "terminated": int(result.terminated),
        "truncated": int(result.truncated),
        "mean_absolute_td_error": diagnostics.mean_absolute_td_error,
        "maximum_absolute_td_error": diagnostics.maximum_absolute_td_error,
        "mean_absolute_q_value": diagnostics.mean_absolute_q_value,
        "maximum_absolute_q_value": diagnostics.maximum_absolute_q_value,
        "mean_action_gap": diagnostics.mean_action_gap,
        "dominant_action_fraction": diagnostics.dominant_action_fraction,
        "action_counts": json.dumps(diagnostics.action_counts),
    }


def print_diagnostic_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    print(
        "    training diagnostics: "
        f"return={mean('episode_return'):.1f}, "
        f"lr={float(rows[-1]['learning_rate']):.3g}, "
        f"|td|={mean('mean_absolute_td_error'):.3g}, "
        f"|Q|={mean('mean_absolute_q_value'):.3g}, "
        f"gap={mean('mean_action_gap'):.3g}, "
        f"dominant action={mean('dominant_action_fraction'):.1%}",
        flush=True,
    )


def run_control(
    environment: str,
    *,
    training_episodes: int,
    evaluation_episodes: int,
    seeds: int,
    learning_rate: float,
    minimum_learning_rate: float,
    discount: float,
    initial_epsilon: float,
    sarsa_final_epsilon: float,
    optimizer_name: str,
    hidden_sizes: tuple[int, ...],
    rollout_steps: int,
    recordings_directory: Path,
    metrics_path: Path,
    diagnostics_path: Path | None,
    validation_episodes: int,
    recording_checkpoints: tuple[int, ...],
    frame_stride: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    if diagnostics_path is not None:
        write_csv(diagnostics_path, DIAGNOSTIC_FIELDS, diagnostic_rows)
    selected_recording_seeds: dict[str, int] = {}

    for algorithm in ALGORITHMS:
        recording_states: dict[int, dict[int, ModelState]] = {}
        recording_agent: Agent | None = None
        for seed in range(seeds):
            print(f"{algorithm}: seed {seed + 1}/{seeds}", flush=True)
            training_env = make_environment(environment)
            evaluation_env = make_environment(environment)
            agent = make_agent(
                algorithm,
                training_env,
                learning_rate=learning_rate,
                discount=discount,
                epsilon=initial_epsilon,
                optimizer_name=optimizer_name,
                hidden_sizes=hidden_sizes,
                seed=seed,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                agent.optimizer,
                T_max=max(1, training_episodes - 1),
                eta_min=minimum_learning_rate,
            )
            completed_episodes = 0
            best_model_state: ModelState | None = None
            best_score: tuple[float, float, float] | None = None
            best_checkpoint = 0
            final_epsilon = algorithm_final_epsilon(
                algorithm,
                initial_epsilon,
                sarsa_final_epsilon,
            )
            for checkpoint in EVALUATION_CHECKPOINTS:
                target_episodes = checkpoint_episode_target(
                    training_episodes,
                    checkpoint,
                )
                print(
                    f"  checkpoint {checkpoint}%: train to {target_episodes}, "
                    f"then evaluate {evaluation_episodes} episodes",
                    flush=True,
                )
                diagnostic_segment_start = len(diagnostic_rows)
                while completed_episodes < target_episodes:
                    agent.epsilon = linearly_decayed_epsilon(
                        initial_epsilon,
                        final_epsilon,
                        completed_episodes,
                        training_episodes,
                    )
                    episode_learning_rate = optimizer_learning_rate(agent)
                    result = train_episode(
                        training_env,
                        agent,
                        seed=seed * training_episodes + completed_episodes,
                        rollout_steps=rollout_steps,
                        collect_diagnostics=diagnostics_path is not None,
                    )
                    scheduler.step()
                    completed_episodes += 1
                    if diagnostics_path is not None:
                        diagnostic_rows.append(
                            diagnostic_row(
                                algorithm,
                                seed,
                                completed_episodes,
                                agent.epsilon,
                                episode_learning_rate,
                                result,
                            )
                        )

                if diagnostics_path is not None:
                    diagnostic_segment = diagnostic_rows[diagnostic_segment_start:]
                    print_diagnostic_summary(diagnostic_segment)
                    write_csv(
                        diagnostics_path,
                        DIAGNOSTIC_FIELDS,
                        diagnostic_rows,
                    )

                current_model_state = copy_model_state(agent.model)
                if checkpoint > 0:
                    validation_results: list[EvaluationResult] = []
                    for validation_episode in range(validation_episodes):
                        result = evaluate_episode(
                            evaluation_env,
                            environment,
                            agent,
                            seed=(500_000 + seed * 100_000 + validation_episode),
                        )
                        validation_results.append(result)
                        rows.append(
                            evaluation_row(
                                algorithm,
                                seed,
                                checkpoint,
                                "validation",
                                checkpoint,
                                validation_episode,
                                result,
                            )
                        )

                    candidate_score = evaluation_score(validation_results)
                    if best_score is None or candidate_score > best_score:
                        best_score = candidate_score
                        best_model_state = copy_model_state(agent.model)
                        best_checkpoint = checkpoint
                    print(
                        "    validation: "
                        f"success={candidate_score[0]:.0%}, "
                        f"return={candidate_score[1]:.1f}, "
                        f"progress={candidate_score[2]:.3f}; "
                        f"selected={best_checkpoint}%",
                        flush=True,
                    )

                if best_model_state is not None:
                    agent.model.load_state_dict(best_model_state)
                    policy_checkpoint = best_checkpoint
                else:
                    policy_checkpoint = 0

                for evaluation_episode in range(evaluation_episodes):
                    evaluation_seed = (
                        1_000_000
                        + seed * 100_000
                        + checkpoint * 1_000
                        + evaluation_episode
                    )
                    result = evaluate_episode(
                        evaluation_env,
                        environment,
                        agent,
                        seed=evaluation_seed,
                    )
                    rows.append(
                        evaluation_row(
                            algorithm,
                            seed,
                            checkpoint,
                            "held_out",
                            policy_checkpoint,
                            evaluation_episode,
                            result,
                        )
                    )
                write_csv(metrics_path, CSV_FIELDS, rows)

                if checkpoint in recording_checkpoints:
                    recording_states.setdefault(seed, {})[checkpoint] = (
                        copy_model_state(agent.model)
                    )
                if checkpoint < 100:
                    agent.model.load_state_dict(current_model_state)
            training_env.close()
            evaluation_env.close()
            recording_agent = agent

        final_seed_scores = {}
        for seed in range(seeds):
            selected = [
                row
                for row in rows
                if row["algorithm"] == algorithm
                and int(row["seed"]) == seed
                and int(row["checkpoint"]) == 100
                and row["evaluation_kind"] == "held_out"
            ]
            final_seed_scores[seed] = (
                float(np.mean([float(row["episode_return"]) for row in selected])),
                float(np.mean([float(row["success"]) for row in selected])),
                float(np.mean([float(row["progress"]) for row in selected])),
            )
        selected_seed = max(final_seed_scores, key=final_seed_scores.__getitem__)
        selected_recording_seeds[algorithm] = selected_seed
        if recording_checkpoints:
            assert recording_agent is not None
            print(
                f"{algorithm}: recordings use best final seed {selected_seed}",
                flush=True,
            )
            for checkpoint in recording_checkpoints:
                recording_agent.model.load_state_dict(
                    recording_states[selected_seed][checkpoint]
                )
                record_evaluation(
                    recordings_directory
                    / f"{algorithm}_checkpoint_{checkpoint:03d}.gif",
                    environment,
                    recording_agent,
                    seed=2_000_000 + checkpoint,
                    frame_stride=frame_stride,
                )
                print(f"  recorded checkpoint {checkpoint}%", flush=True)
    return rows, selected_recording_seeds


def aggregate_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    aggregate: list[dict[str, object]] = []
    metrics = (
        "episode_return",
        "episode_length",
        "success",
        "truncated",
        "progress",
    )
    for algorithm in ALGORITHMS:
        for checkpoint in EVALUATION_CHECKPOINTS:
            selected = [
                row
                for row in rows
                if row["algorithm"] == algorithm
                and row["checkpoint"] == checkpoint
                and row["evaluation_kind"] == "held_out"
            ]
            seed_means: dict[str, list[float]] = {metric: [] for metric in metrics}
            for seed in sorted({int(row["seed"]) for row in selected}):
                seed_rows = [row for row in selected if row["seed"] == seed]
                for metric in metrics:
                    seed_means[metric].append(
                        float(np.mean([float(row[metric]) for row in seed_rows]))
                    )

            result: dict[str, object] = {
                "algorithm": algorithm,
                "checkpoint": checkpoint,
                "seeds": len(seed_means["success"]),
                "policy_checkpoints": sorted(
                    {int(row["policy_checkpoint"]) for row in selected}
                ),
            }
            for metric, values_list in seed_means.items():
                values = np.asarray(values_list, dtype=float)
                result[metric] = float(np.mean(values))
                result[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
            aggregate.append(result)
    return aggregate


def write_figure(
    path: Path,
    environment: str,
    aggregate: list[dict[str, object]],
) -> None:
    progress_label, progress_goal, _ = progress_configuration(environment)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for algorithm in ALGORITHMS:
        selected = [row for row in aggregate if row["algorithm"] == algorithm]
        checkpoints = np.asarray([row["checkpoint"] for row in selected], dtype=float)
        for axis, metric in zip(axes, ("success", "progress"), strict=True):
            means = np.asarray([row[metric] for row in selected], dtype=float)
            standard_errors = np.asarray(
                [
                    float(row[f"{metric}_std"]) / np.sqrt(float(row["seeds"]))
                    for row in selected
                ]
            )
            (line,) = axis.plot(checkpoints, means, marker="o", label=algorithm)
            axis.fill_between(
                checkpoints,
                means - 1.96 * standard_errors,
                means + 1.96 * standard_errors,
                color=line.get_color(),
                alpha=0.12,
            )

    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel(progress_label)
    axes[1].axhline(
        progress_goal,
        color="black",
        linestyle="--",
        linewidth=1,
        label="termination threshold",
    )
    for axis in axes:
        axis.set_xlabel("Training completed (%)")
        axis.set_xticks(EVALUATION_CHECKPOINTS)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Best validated greedy policy on held-out episodes")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def mean_and_std(row: dict[str, object], metric: str) -> str:
    return f"{float(row[metric]):.3f} +/- {float(row[f'{metric}_std']):.3f}"


def write_outputs(
    output: Path,
    environment: str,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    write_csv(output / "metrics.csv", CSV_FIELDS, rows)
    write_metadata(output / "metadata.json", metadata)

    aggregate = aggregate_rows(rows)
    write_figure(
        output / "figures" / "checkpoint_evaluation.png",
        environment,
        aggregate,
    )
    progress_label, _, _ = progress_configuration(environment)
    table_rows = tuple(
        (
            row["algorithm"],
            f"{row['checkpoint']}%",
            ", ".join(f"{value}%" for value in row["policy_checkpoints"]),
            mean_and_std(row, "success"),
            mean_and_std(row, "progress"),
            mean_and_std(row, "episode_length"),
            mean_and_std(row, "truncated"),
        )
        for row in aggregate
    )
    final_seed_rows = []
    for algorithm in ALGORITHMS:
        for seed in sorted({int(row["seed"]) for row in rows}):
            selected = [
                row
                for row in rows
                if row["algorithm"] == algorithm
                and int(row["seed"]) == seed
                and int(row["checkpoint"]) == 100
                and row["evaluation_kind"] == "held_out"
            ]
            if selected:
                mean_return = np.mean(
                    [float(row["episode_return"]) for row in selected]
                )
                mean_success = np.mean([float(row["success"]) for row in selected])
                mean_progress = np.mean([float(row["progress"]) for row in selected])
                mean_length = np.mean(
                    [float(row["episode_length"]) for row in selected]
                )
                final_seed_rows.append(
                    (
                        algorithm,
                        seed,
                        f"{mean_return:.3f}",
                        f"{mean_success:.3f}",
                        f"{mean_progress:.3f}",
                        f"{mean_length:.2f}",
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
                0 if item.stem.startswith("sarsa") else 1,
                item.name,
            ),
        )
    )
    write_summary(
        output / "summary.html",
        title=f"Function approximation on {environment}",
        metadata={
            "Environment": environment,
            "Preset": metadata["preset"],
            "Training episodes": metadata["training_episodes"],
            "Evaluation": (
                f"{metadata['evaluation_episodes']} episodes per checkpoint and seed"
            ),
            "Model selection": (
                f"best checkpoint on {metadata['validation_episodes']} fixed-seed "
                "validation episodes"
            ),
            "Seeds": metadata["seeds"],
            "Network": f"MLP with hidden sizes {metadata['hidden_sizes']}",
            "Optimizer": (
                f"{metadata['optimizer']}, cosine learning rate "
                f"{metadata['learning_rate']} to "
                f"{metadata['minimum_learning_rate']}"
            ),
            "Discount": metadata["discount"],
            "Rollout length": metadata["rollout_steps"],
            "SARSA epsilon": (
                str(metadata["initial_epsilon"])
                if metadata["sarsa_epsilon_schedule"] == "constant"
                else (
                    f"{metadata['initial_epsilon']} to "
                    f"{metadata['sarsa_final_epsilon']} (linear)"
                )
            ),
            "Q-learning epsilon": f"{metadata['initial_epsilon']} (constant)",
            "Observations": "statically rescaled to [-1, 1]",
            "Recorded seed by algorithm": metadata["recording_seed_by_algorithm"],
        },
        tables=(
            SummaryTable(
                "Best validated policy on held-out episodes",
                (
                    "Algorithm",
                    "Training budget",
                    "Selected checkpoint",
                    "Success rate",
                    progress_label,
                    "Episode length",
                    "Truncation rate",
                ),
                table_rows,
            ),
            SummaryTable(
                "Final held-out evaluation by seed",
                (
                    "Algorithm",
                    "Seed",
                    "Mean return",
                    "Success rate",
                    progress_label,
                    "Episode length",
                ),
                tuple(final_seed_rows),
            ),
        ),
        figures=(
            SummaryMedia(
                "Success and environment progress",
                Path("figures/checkpoint_evaluation.png"),
            ),
        ),
        recordings=recordings,
    )


def experiment_defaults(environment: str, preset: str) -> dict[str, int]:
    if preset == "quick":
        return {
            "training_episodes": 10 if environment == "MountainCar-v0" else 5,
            "evaluation_episodes": 2,
            "seeds": 1,
        }
    if preset == "tuning":
        return {
            "training_episodes": 500 if environment == "MountainCar-v0" else 250,
            "evaluation_episodes": 10,
            "seeds": 1,
        }
    return {
        "training_episodes": 2_000 if environment == "MountainCar-v0" else 1_000,
        "evaluation_episodes": 50,
        "seeds": 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        "--env",
        "-e",
        choices=ENVIRONMENTS,
        default=ENVIRONMENTS[0],
    )
    parser.add_argument(
        "--preset",
        "-p",
        choices=("quick", "tuning", "standard"),
        default="standard",
    )
    parser.add_argument("--training-episodes", "--train", type=int)
    parser.add_argument("--evaluation-episodes", "--eval", type=int)
    parser.add_argument("--seeds", "-n", type=int)
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.001)
    parser.add_argument(
        "--minimum-learning-rate",
        "--lr-min",
        type=float,
        help="minimum cosine learning rate (default: 1%% of --lr)",
    )
    parser.add_argument(
        "--validation-episodes",
        "--val",
        type=int,
        help="fixed-seed episodes for best-checkpoint selection (default: up to 5)",
    )
    parser.add_argument("--discount", "--gamma", type=float, default=0.99)
    parser.add_argument(
        "--rollout-steps",
        "--rollout",
        "--n-steps",
        "--n-step",
        dest="rollout_steps",
        type=int,
        default=1,
        help="maximum transitions per semi-gradient TD rollout (default: 1)",
    )
    parser.add_argument("--epsilon", "--eps", type=float, default=0.1)
    parser.add_argument(
        "--sarsa-final-epsilon",
        "--sarsa-eps-final",
        "--final-epsilon",
        "--eps-final",
        type=float,
        help="final SARSA epsilon (default: min(0.01, --eps))",
    )
    parser.add_argument(
        "--optimizer",
        "--opt",
        "-o",
        choices=("sgd", "adam"),
        default="adam",
    )
    parser.add_argument(
        "--hidden-sizes",
        "--hidden",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_SIZES,
        metavar="N",
    )
    parser.add_argument(
        "--recordings",
        "--record",
        "-r",
        choices=("none", "final", "checkpoints"),
    )
    parser.add_argument(
        "--diagnostics",
        "--diag",
        action="store_true",
        help="write per-training-episode value-learning diagnostics",
    )
    args = parser.parse_args()

    defaults = experiment_defaults(args.environment, args.preset)
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
    validation_episodes = (
        min(5, evaluation_episodes)
        if args.validation_episodes is None
        else args.validation_episodes
    )
    if min(training_episodes, evaluation_episodes, validation_episodes, seeds) < 1:
        parser.error(
            "training, validation, evaluation episodes, and seeds must be positive"
        )
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    minimum_learning_rate = (
        args.learning_rate * 0.01
        if args.minimum_learning_rate is None
        else args.minimum_learning_rate
    )
    if (
        not np.isfinite(minimum_learning_rate)
        or minimum_learning_rate < 0
        or minimum_learning_rate > args.learning_rate
    ):
        parser.error("minimum learning rate must be finite and between 0 and --lr")
    if not np.isfinite(args.discount) or not 0 <= args.discount <= 1:
        parser.error("discount must be finite and in [0, 1]")
    if args.rollout_steps < 1:
        parser.error("rollout steps must be positive")
    if not np.isfinite(args.epsilon) or not 0 <= args.epsilon <= 1:
        parser.error("epsilon must be finite and in [0, 1]")
    sarsa_final_epsilon = (
        min(0.01, args.epsilon)
        if args.sarsa_final_epsilon is None
        else args.sarsa_final_epsilon
    )
    if not np.isfinite(sarsa_final_epsilon) or not 0 <= sarsa_final_epsilon <= 1:
        parser.error("final SARSA epsilon must be finite and in [0, 1]")
    if sarsa_final_epsilon > args.epsilon:
        parser.error("final SARSA epsilon must not exceed initial epsilon")
    if any(hidden_size <= 0 for hidden_size in args.hidden_sizes):
        parser.error("hidden sizes must be positive")

    hidden_sizes = tuple(args.hidden_sizes)
    recording_mode = args.recordings
    if recording_mode is None:
        recording_mode = {
            "quick": "final",
            "tuning": "none",
            "standard": "checkpoints",
        }[args.preset]
    recording_checkpoints = {
        "none": (),
        "final": (100,),
        "checkpoints": RECORDING_CHECKPOINTS,
    }[recording_mode]

    inspection_env = make_environment(args.environment)
    assert isinstance(inspection_env.observation_space, gym.spaces.Box)
    assert isinstance(inspection_env.action_space, gym.spaces.Discrete)
    original_space = inspection_env.unwrapped.observation_space
    assert isinstance(original_space, gym.spaces.Box)
    progress_label, progress_goal, success_definition = progress_configuration(
        args.environment
    )
    frame_stride = 2 if args.environment == "MountainCar-v0" else 4
    max_episode_steps = (
        inspection_env.spec.max_episode_steps
        if inspection_env.spec is not None
        else None
    )
    observation_shape = list(inspection_env.observation_space.shape)
    number_of_actions = int(inspection_env.action_space.n)
    inspection_env.close()

    output = create_run_directory("function_approximation", args.environment)
    (output / "figures").mkdir()
    recordings_directory = output / "recordings"
    if recording_checkpoints:
        recordings_directory.mkdir()
    metadata: dict[str, object] = {
        "status": "running",
        "environment": args.environment,
        "preset": args.preset,
        "algorithms": list(ALGORITHMS),
        "training_episodes": training_episodes,
        "evaluation_episodes": evaluation_episodes,
        "seeds": seeds,
        "observation_shape": observation_shape,
        "original_observation_low": original_space.low.tolist(),
        "original_observation_high": original_space.high.tolist(),
        "observation_scaling": [-1.0, 1.0],
        "number_of_actions": number_of_actions,
        "hidden_sizes": list(hidden_sizes),
        "optimizer": "SGD" if args.optimizer == "sgd" else "Adam",
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": minimum_learning_rate,
        "learning_rate_schedule": "cosine annealing once per training episode",
        "discount": args.discount,
        "rollout_steps": args.rollout_steps,
        "initial_epsilon": args.epsilon,
        "sarsa_final_epsilon": sarsa_final_epsilon,
        "sarsa_epsilon_schedule": (
            "constant"
            if sarsa_final_epsilon == args.epsilon
            else "linear over training episodes"
        ),
        "q_learning_final_epsilon": args.epsilon,
        "q_learning_epsilon_schedule": "constant",
        "max_episode_steps": max_episode_steps,
        "evaluation_checkpoints": list(EVALUATION_CHECKPOINTS),
        "validation_episodes": validation_episodes,
        "validation_environment_seed": (
            "500000 + seed * 100000 + validation_episode; fixed across checkpoints"
        ),
        "validation_selection": (
            "lexicographic maximum of mean success, return, then progress"
        ),
        "reported_checkpoint_policy": (
            "best validation checkpoint at or before each training budget"
        ),
        "recording_checkpoints": list(recording_checkpoints),
        "recording_mode": recording_mode,
        "recording_environment_seed": (
            "2000000 + checkpoint" if recording_checkpoints else None
        ),
        "recording_frame_stride": frame_stride if recording_checkpoints else None,
        "training_diagnostics": args.diagnostics,
        "training_diagnostics_file": (
            "training_diagnostics.csv" if args.diagnostics else None
        ),
        "diagnostic_action_values": (
            "all action values before each update" if args.diagnostics else None
        ),
        "training_environment_seed": "seed * training_episodes + episode",
        "evaluation_environment_seed": (
            "1000000 + seed * 100000 + checkpoint * 1000 + evaluation_episode"
        ),
        "evaluation_policy": "frozen greedy with deterministic first-index tie break",
        "success_definition": success_definition,
        "progress_metric": progress_label,
        "progress_goal": progress_goal,
        "truncation_target": "bootstrap then stop interaction",
        "variability": "standard deviation across seed evaluation means",
        "confidence_interval": "mean +/- 1.96 * standard error",
        "smoothing_window": 1,
        "versions": {
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    write_metadata(output / "metadata.json", metadata)
    rows, selected_recording_seeds = run_control(
        args.environment,
        training_episodes=training_episodes,
        evaluation_episodes=evaluation_episodes,
        seeds=seeds,
        learning_rate=args.learning_rate,
        minimum_learning_rate=minimum_learning_rate,
        discount=args.discount,
        initial_epsilon=args.epsilon,
        sarsa_final_epsilon=sarsa_final_epsilon,
        optimizer_name=args.optimizer,
        hidden_sizes=hidden_sizes,
        rollout_steps=args.rollout_steps,
        recordings_directory=recordings_directory,
        metrics_path=output / "metrics.csv",
        diagnostics_path=(
            output / "training_diagnostics.csv" if args.diagnostics else None
        ),
        validation_episodes=validation_episodes,
        recording_checkpoints=recording_checkpoints,
        frame_stride=frame_stride,
    )
    metadata["recording_seed_selection"] = (
        "highest final mean held-out return; success and progress break ties"
    )
    metadata["recording_seed_by_algorithm"] = selected_recording_seeds
    write_outputs(output, args.environment, rows, metadata)
    metadata["status"] = "complete"
    write_metadata(output / "metadata.json", metadata)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
