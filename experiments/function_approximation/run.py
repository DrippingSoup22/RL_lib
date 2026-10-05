"""Run semi-gradient prediction and control on Gymnasium environments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from experiments.common import (
    annotated_frame,
    checkpoint_episode_target,
    create_run_directory,
    write_csv,
    write_metadata,
)
from experiments.function_approximation.configuration import (
    initial_metadata,
    parse_config,
)
from experiments.function_approximation.diagnostics import (
    EpisodeDiagnostics,
    EpisodeDiagnosticTracker,
)
from experiments.function_approximation.environments import (
    action_name,
    episode_progress,
    episode_succeeded,
    make_environment,
    progress_value,
)
from experiments.function_approximation.report import (
    CONTROL_CSV_FIELDS,
    PREDICTION_CSV_FIELDS,
    write_control_report,
    write_prediction_report,
)
from rl_lib.algorithms.function_approximation import (
    SemiGradientQLearning,
    SemiGradientSARSA,
    SemiGradientTDPrediction,
)
from rl_lib.networks import ActionValueNetwork, StateValueNetwork
from rl_lib.trajectories import EpisodeStep, discounted_returns

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


def make_prediction_agent(
    env: gym.Env,
    *,
    learning_rate: float,
    discount: float,
    optimizer_name: str,
    hidden_sizes: tuple[int, ...],
    seed: int,
) -> SemiGradientTDPrediction:
    """Create a state-value estimator for the fixed prediction policy."""
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("environment must have a Box observation space")

    observation_size = int(np.prod(env.observation_space.shape))
    torch.manual_seed(seed)
    model = StateValueNetwork(observation_size, hidden_sizes=hidden_sizes)
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    return SemiGradientTDPrediction(model, optimizer, discount=discount)


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


def train_prediction_episode(
    env: gym.Env,
    predictor: SemiGradientTDPrediction,
    *,
    environment_seed: int,
    action_seed: int,
    rollout_steps: int,
) -> EpisodeResult:
    """Train the value estimator for one episode of the fixed policy."""
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("environment must have a Discrete action space")
    number_of_actions = int(env.action_space.n)
    rng = np.random.default_rng(action_seed)
    observation, _ = env.reset(seed=environment_seed)
    rollout: list[EpisodeStep[np.ndarray]] = []
    episode_return = 0.0
    episode_length = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.integers(number_of_actions))
        next_observation, reward, terminated, truncated, _ = env.step(action)
        rollout.append(
            EpisodeStep(
                np.asarray(observation, dtype=np.float32).copy(),
                action,
                float(reward),
            )
        )
        episode_return += float(reward)
        episode_length += 1
        if len(rollout) == rollout_steps or terminated or truncated:
            predictor.update(
                tuple(rollout),
                next_observation,
                terminated=terminated,
            )
            rollout.clear()
        observation = next_observation

    return EpisodeResult(
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=terminated,
        truncated=truncated,
    )


def evaluate_prediction_episode(
    env: gym.Env,
    predictor: SemiGradientTDPrediction,
    *,
    environment_seed: int,
    action_seed: int,
) -> dict[str, float | int]:
    """Compare frozen predictions with held-out Monte Carlo returns."""
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("environment must have a Discrete action space")
    number_of_actions = int(env.action_space.n)
    rng = np.random.default_rng(action_seed)
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    observation, _ = env.reset(seed=environment_seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.integers(number_of_actions))
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))

    was_training = predictor.model.training
    predictor.model.eval()
    try:
        with torch.no_grad():
            predictions = (
                predictor.model(
                    torch.as_tensor(np.asarray(observations), dtype=torch.float32)
                )
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
    finally:
        predictor.model.train(was_training)
    targets = discounted_returns(rewards, predictor.discount)
    errors = predictions - targets
    return {
        "episode_return": float(np.sum(rewards)),
        "episode_length": len(rewards),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "root_mean_squared_error": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_error": float(np.mean(errors)),
    }


def model_action_values(agent: Agent, observation: np.ndarray) -> np.ndarray:
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        action_values = agent.model(observation_tensor)
    return action_values.detach().cpu().numpy()


def greedy_action(agent: Agent, observation: np.ndarray) -> int:
    return int(np.argmax(model_action_values(agent, observation)))


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
        maximum_progress = progress_value(environment, observation, 0)
        episode_return = 0.0
        episode_length = 0
        final_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = greedy_action(agent, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            final_reward = float(reward)
            episode_length += 1
            maximum_progress = max(
                maximum_progress,
                progress_value(environment, observation, episode_length),
            )
    finally:
        agent.model.train(was_training)

    return EvaluationResult(
        episode_return=episode_return,
        episode_length=episode_length,
        success=episode_succeeded(
            environment,
            terminated=terminated,
            truncated=truncated,
            final_reward=final_reward,
        ),
        terminated=terminated,
        truncated=truncated,
        progress=episode_progress(environment, maximum_progress, episode_return),
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


def run_prediction(
    environment: str,
    *,
    training_episodes: int,
    evaluation_episodes: int,
    seed_values: tuple[int, ...],
    learning_rate: float,
    minimum_learning_rate: float,
    discount: float,
    optimizer_name: str,
    hidden_sizes: tuple[int, ...],
    rollout_steps: int,
    metrics_path: Path | None,
    model_path: Path | None,
    checkpoints: tuple[int, ...],
) -> tuple[list[dict[str, object]], int]:
    """Train and evaluate semi-gradient TD prediction under a fixed policy."""
    rows: list[dict[str, object]] = []
    final_model_states: dict[int, ModelState] = {}
    for trial_index, seed in enumerate(seed_values):
        print(
            f"td_prediction: seed trial {trial_index + 1}/{len(seed_values)} "
            f"(seed={seed})",
            flush=True,
        )
        training_env = make_environment(environment)
        evaluation_env = make_environment(environment)
        predictor = make_prediction_agent(
            training_env,
            learning_rate=learning_rate,
            discount=discount,
            optimizer_name=optimizer_name,
            hidden_sizes=hidden_sizes,
            seed=seed,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            predictor.optimizer,
            T_max=max(1, training_episodes - 1),
            eta_min=minimum_learning_rate,
        )
        completed_episodes = 0
        for checkpoint in checkpoints:
            target_episodes = checkpoint_episode_target(
                training_episodes,
                checkpoint,
            )
            print(
                f"  checkpoint {checkpoint}%: train to {target_episodes}, "
                f"then evaluate {evaluation_episodes} episodes",
                flush=True,
            )
            while completed_episodes < target_episodes:
                train_prediction_episode(
                    training_env,
                    predictor,
                    environment_seed=(seed * training_episodes + completed_episodes),
                    action_seed=(
                        10_000_000 + seed * training_episodes + completed_episodes
                    ),
                    rollout_steps=rollout_steps,
                )
                scheduler.step()
                completed_episodes += 1
            checkpoint_rows = []
            for evaluation_episode in range(evaluation_episodes):
                result = {
                    "algorithm": "td_prediction",
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "evaluation_episode": evaluation_episode,
                    **evaluate_prediction_episode(
                        evaluation_env,
                        predictor,
                        environment_seed=(
                            1_000_000
                            + seed * 100_000
                            + checkpoint * 1_000
                            + evaluation_episode
                        ),
                        action_seed=(
                            11_000_000
                            + seed * 100_000
                            + checkpoint * 1_000
                            + evaluation_episode
                        ),
                    ),
                }
                checkpoint_rows.append(result)
                rows.append(result)
            mean_rmse = float(
                np.mean(
                    [float(row["root_mean_squared_error"]) for row in checkpoint_rows]
                )
            )
            print(f"    held-out rmse={mean_rmse:.3f}", flush=True)
            if metrics_path is not None:
                write_csv(metrics_path, PREDICTION_CSV_FIELDS, rows)

        final_model_states[seed] = copy_model_state(predictor.model)
        training_env.close()
        evaluation_env.close()

    final_scores = {}
    for seed in seed_values:
        selected = [
            row
            for row in rows
            if int(row["seed"]) == seed and int(row["checkpoint"]) == 100
        ]
        final_scores[seed] = float(
            np.mean([float(row["root_mean_squared_error"]) for row in selected])
        )
    selected_seed = min(final_scores, key=final_scores.__getitem__)
    if model_path is not None:
        torch.save(
            {
                "algorithm": "td_prediction",
                "seed": selected_seed,
                "model_state_dict": final_model_states[selected_seed],
            },
            model_path,
        )
    return rows, selected_seed


def run_control(
    environment: str,
    selected_algorithm: str,
    *,
    training_episodes: int,
    evaluation_episodes: int,
    seed_values: tuple[int, ...],
    learning_rate: float,
    minimum_learning_rate: float,
    discount: float,
    initial_epsilon: float,
    sarsa_final_epsilon: float,
    optimizer_name: str,
    hidden_sizes: tuple[int, ...],
    rollout_steps: int,
    recordings_directory: Path | None,
    metrics_path: Path | None,
    diagnostics_path: Path | None,
    model_path: Path | None,
    validation_episodes: int,
    recording_checkpoints: tuple[int, ...],
    checkpoints: tuple[int, ...],
    frame_stride: int,
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    if diagnostics_path is not None:
        write_csv(diagnostics_path, DIAGNOSTIC_FIELDS, diagnostic_rows)
    algorithm = selected_algorithm
    recording_states: dict[int, dict[int, ModelState]] = {}
    selected_policy_checkpoints: dict[int, int] = {}
    recording_agent: Agent | None = None
    for trial_index, seed in enumerate(seed_values):
        print(
            f"{algorithm}: seed trial {trial_index + 1}/{len(seed_values)} "
            f"(seed={seed})",
            flush=True,
        )
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
        for checkpoint in checkpoints:
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
                    1_000_000 + seed * 100_000 + checkpoint * 1_000 + evaluation_episode
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
            if metrics_path is not None:
                write_csv(metrics_path, CONTROL_CSV_FIELDS, rows)

            if checkpoint in recording_checkpoints or (
                model_path is not None and checkpoint == 100
            ):
                recording_states.setdefault(seed, {})[checkpoint] = copy_model_state(
                    agent.model
                )
            if checkpoint < 100:
                agent.model.load_state_dict(current_model_state)
        selected_policy_checkpoints[seed] = best_checkpoint
        training_env.close()
        evaluation_env.close()
        recording_agent = agent

    final_seed_scores = {}
    for seed in seed_values:
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
    if model_path is not None:
        torch.save(
            {
                "algorithm": algorithm,
                "seed": selected_seed,
                "policy_checkpoint": selected_policy_checkpoints[selected_seed],
                "model_state_dict": recording_states[selected_seed][100],
            },
            model_path,
        )
    if recording_checkpoints:
        assert recording_agent is not None
        assert recordings_directory is not None
        print(
            f"{algorithm}: recordings use best final seed {selected_seed}",
            flush=True,
        )
        for checkpoint in recording_checkpoints:
            recording_agent.model.load_state_dict(
                recording_states[selected_seed][checkpoint]
            )
            record_evaluation(
                recordings_directory / f"{algorithm}_checkpoint_{checkpoint:03d}.gif",
                environment,
                recording_agent,
                seed=2_000_000 + checkpoint,
                frame_stride=frame_stride,
            )
            print(f"  recorded checkpoint {checkpoint}%", flush=True)
    return rows, selected_seed


def main() -> None:
    config = parse_config()
    output = (
        None
        if config.preset == "quick"
        else create_run_directory(
            "function_approximation",
            config.environment,
            config.algorithm,
            config.preset,
        )
    )
    recordings_directory = output / "recordings" if output is not None else None
    if output is not None:
        (output / "figures").mkdir()
        if config.recording_checkpoints:
            assert recordings_directory is not None
            recordings_directory.mkdir()

    metadata = initial_metadata(config)
    if output is not None:
        write_metadata(output / "metadata.json", metadata)
    metrics_path = output / "metrics.csv" if output is not None else None
    diagnostics_path = (
        output / "training_diagnostics.csv"
        if config.diagnostics and output is not None
        else None
    )
    model_path = (
        output / "best_model.pt"
        if output is not None and config.preset == "standard"
        else None
    )
    if config.algorithm == "td_prediction":
        rows, selected_seed = run_prediction(
            config.environment,
            training_episodes=config.training_episodes,
            evaluation_episodes=config.evaluation_episodes,
            seed_values=config.seed_values,
            learning_rate=config.learning_rate,
            minimum_learning_rate=config.minimum_learning_rate,
            discount=config.discount,
            optimizer_name=config.optimizer,
            hidden_sizes=config.hidden_sizes,
            rollout_steps=config.rollout_steps,
            metrics_path=metrics_path,
            model_path=model_path,
            checkpoints=config.checkpoints,
        )
    else:
        rows, selected_seed = run_control(
            config.environment,
            config.algorithm,
            training_episodes=config.training_episodes,
            evaluation_episodes=config.evaluation_episodes,
            seed_values=config.seed_values,
            learning_rate=config.learning_rate,
            minimum_learning_rate=config.minimum_learning_rate,
            discount=config.discount,
            initial_epsilon=config.epsilon,
            sarsa_final_epsilon=config.sarsa_final_epsilon,
            optimizer_name=config.optimizer,
            hidden_sizes=config.hidden_sizes,
            rollout_steps=config.rollout_steps,
            recordings_directory=recordings_directory,
            metrics_path=metrics_path,
            diagnostics_path=diagnostics_path,
            model_path=model_path,
            validation_episodes=config.validation_episodes,
            recording_checkpoints=config.recording_checkpoints,
            checkpoints=config.checkpoints,
            frame_stride=config.frame_stride,
        )
    if output is None:
        print(
            "Quick compatibility run complete; no artifacts were written.",
            flush=True,
        )
        return

    metadata["recording_seed_selection"] = (
        "highest final mean held-out return; success and progress break ties"
        if config.recording_checkpoints
        else None
    )
    metadata["selected_seed"] = (
        selected_seed
        if config.algorithm == "td_prediction" or config.recording_checkpoints
        else None
    )
    metadata["model_file"] = "best_model.pt" if config.preset == "standard" else None
    metadata["model_seed"] = selected_seed if config.preset == "standard" else None
    if config.algorithm == "td_prediction":
        write_prediction_report(output, rows, metadata)
    else:
        write_control_report(output, rows, metadata)
    metadata["status"] = "complete"
    write_metadata(output / "metadata.json", metadata)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
