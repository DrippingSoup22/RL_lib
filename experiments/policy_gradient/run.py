"""Run one policy-gradient algorithm on a Gymnasium environment."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
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
    evaluation_checkpoints,
    write_csv,
    write_metadata,
)
from experiments.policy_gradient.configuration import (
    ALGORITHM_LABELS,
    ALGORITHMS,
    ExperimentConfig,
    initial_metadata,
    parse_config,
)
from experiments.policy_gradient.environments import (
    action_name,
    episode_succeeded,
    make_environment,
    recording_frame_stride,
)
from experiments.policy_gradient.report import CSV_FIELDS, write_report
from experiments.policy_gradient.runners.a2c import (
    train_episode as train_a2c_episode,
)
from experiments.policy_gradient.runners.a3c import (
    train_episodes as train_a3c_episodes,
)
from experiments.policy_gradient.runners.common import (
    TrainingEpisodeResult,
    generate_episode,
    observation_array,
)
from experiments.policy_gradient.runners.ppo import PPOTrainingBatchResult
from experiments.policy_gradient.runners.ppo import train_episodes as train_ppo_episodes
from experiments.policy_gradient.runners.reinforce import (
    train_episode as train_reinforce_episode,
)
from rl_lib.algorithms.policy_gradient import A2C, PPO, Reinforce, ReinforceWithBaseline
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork

DIAGNOSTIC_FIELDS = (
    "algorithm",
    "seed",
    "training_episode",
    "actor_learning_rate",
    "critic_learning_rate",
    "episode_return",
    "episode_length",
    "terminated",
    "truncated",
    "updates",
    "actor_loss",
    "critic_loss",
    "policy_entropy",
)
Agent = Reinforce | ReinforceWithBaseline | A2C | PPO
ModelState = dict[str, torch.Tensor]


@dataclass(frozen=True)
class EvaluationResult:
    episode_return: float
    episode_length: int
    success: bool
    terminated: bool
    truncated: bool


def make_optimizer(
    name: str,
    parameters: Iterable[torch.Tensor],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Construct one of the optimizers exposed by the runner."""
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer must be 'sgd', 'adam', or 'adamw'")


def make_agent(
    algorithm: str,
    env: gym.Env,
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    discount: float,
    entropy_coefficient: float,
    hidden_sizes: tuple[int, ...],
    seed: int,
    ppo_clip_ratio: float = 0.2,
) -> Agent:
    """Build an agent, or the shared actor-critic state used by A3C."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("environment must have a Box observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("environment must have a Discrete action space")

    observation_size = int(np.prod(env.observation_space.shape))
    number_of_actions = int(env.action_space.n)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor_model = DiscretePolicyNetwork(
            observation_size,
            number_of_actions,
            hidden_sizes,
        )
        actor_optimizer = make_optimizer(
            optimizer_name,
            actor_model.parameters(),
            actor_learning_rate,
            weight_decay,
        )
        if algorithm == "reinforce":
            return Reinforce(actor_model, actor_optimizer, discount)

        critic_model = StateValueNetwork(observation_size, hidden_sizes)
        critic_optimizer = make_optimizer(
            optimizer_name,
            critic_model.parameters(),
            critic_learning_rate,
            weight_decay,
        )
        if algorithm == "reinforce_with_baseline":
            return ReinforceWithBaseline(
                actor_model,
                actor_optimizer,
                critic_model,
                critic_optimizer,
                discount,
            )
        if algorithm == "ppo":
            return PPO(
                actor_model,
                actor_optimizer,
                critic_model,
                critic_optimizer,
                clip_ratio=ppo_clip_ratio,
                entropy_coefficient=entropy_coefficient,
                seed=seed,
            )
        if algorithm not in ("a2c", "a3c"):
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        # A3C workers wrap these shared models in A3C. The parent keeps this
        # synchronous container only for frozen evaluation and recordings.
        return A2C(
            actor_model,
            actor_optimizer,
            critic_model,
            critic_optimizer,
            discount,
            entropy_coefficient,
        )


def make_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    training_episodes: int,
    minimum_learning_rate: float,
    warmup_episodes: int,
    warmup_start_factor: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Warm up linearly, then anneal to the minimum with one cosine cycle."""
    if len(optimizer.param_groups) != 1:
        raise ValueError("policy-gradient optimizers need one parameter group")
    peak_learning_rate = float(optimizer.param_groups[0]["lr"])
    minimum_factor = minimum_learning_rate / peak_learning_rate
    cosine_episodes = max(1, training_episodes - warmup_episodes)

    def learning_rate_factor(episode: int) -> float:
        if warmup_episodes and episode <= warmup_episodes:
            progress = episode / warmup_episodes
            return warmup_start_factor + progress * (1.0 - warmup_start_factor)

        progress = (episode - warmup_episodes) / cosine_episodes
        return minimum_factor + 0.5 * (1.0 - minimum_factor) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)


def train_episode(
    env: gym.Env,
    agent: Agent,
    *,
    rollout_steps: int,
    environment_seed: int,
    action_seed: int,
    collect_diagnostics: bool,
) -> TrainingEpisodeResult:
    """Train one episode using the update cadence owned by the algorithm."""
    if isinstance(agent, A2C):
        return train_a2c_episode(
            env,
            agent,
            rollout_steps=rollout_steps,
            environment_seed=environment_seed,
            action_seed=action_seed,
            collect_diagnostics=collect_diagnostics,
        )

    if isinstance(agent, PPO):
        raise TypeError("PPO must train a batch of episodes with train_ppo_episodes")

    return train_reinforce_episode(
        env,
        agent,
        environment_seed=environment_seed,
        action_seed=action_seed,
        collect_diagnostics=collect_diagnostics,
    )


def evaluate_episode(
    env: gym.Env,
    environment: str,
    agent: Agent,
    *,
    environment_seed: int,
    action_seed: int,
) -> EvaluationResult:
    """Evaluate the frozen stochastic actor for one episode."""
    was_training = agent.actor_model.training
    agent.actor_model.eval()
    try:
        episode = generate_episode(
            env,
            agent,
            environment_seed=environment_seed,
            action_seed=action_seed,
        )
    finally:
        agent.actor_model.train(was_training)

    return EvaluationResult(
        episode_return=sum(step.reward for step in episode.steps),
        episode_length=len(episode.steps),
        success=episode_succeeded(
            environment,
            terminated=episode.terminated,
            truncated=episode.truncated,
            final_reward=episode.steps[-1].reward,
        ),
        terminated=episode.terminated,
        truncated=episode.truncated,
    )


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
    environment_seed: int,
    action_seed: int,
    frame_stride: int,
) -> None:
    """Record one frozen stochastic-policy episode as an annotated GIF."""
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_environment(environment, render_mode="rgb_array")
    was_training = agent.actor_model.training
    agent.actor_model.eval()
    frames: list[Image.Image] = []
    durations: list[int] = []
    try:
        observation, _ = env.reset(seed=environment_seed)
        state = observation_array(observation, agent.actor_model.observation_size)
        frames.append(rendered_frame(env, "Frozen stochastic evaluation: start"))
        durations.append(100)
        terminated = truncated = False
        step = 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_seed)
            while not (terminated or truncated):
                action = agent.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action)
                state = observation_array(
                    observation,
                    agent.actor_model.observation_size,
                )
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
        agent.actor_model.train(was_training)
        env.close()

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


def evaluation_row(
    algorithm: str,
    seed: int,
    checkpoint: int,
    evaluation_episode: int,
    result: EvaluationResult,
) -> dict[str, object]:
    """Convert one frozen-policy evaluation to the raw metrics schema."""
    return {
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint": checkpoint,
        "evaluation_episode": evaluation_episode,
        "episode_return": result.episode_return,
        "episode_length": result.episode_length,
        "success": int(result.success),
        "terminated": int(result.terminated),
        "truncated": int(result.truncated),
    }


def optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate from an optimizer with one parameter group."""
    if len(optimizer.param_groups) != 1:
        raise ValueError("policy-gradient optimizers need one parameter group")
    return float(optimizer.param_groups[0]["lr"])


def critic_optimizer(agent: Agent) -> torch.optim.Optimizer | None:
    """Return the critic optimizer used by baseline and actor-critic agents."""
    if isinstance(agent, (ReinforceWithBaseline, A2C, PPO)):
        return agent.critic_optimizer
    return None


def ppo_episode_training_results(
    batch: PPOTrainingBatchResult,
    *,
    collect_diagnostics: bool,
) -> tuple[TrainingEpisodeResult, ...]:
    """Attach batch-level PPO diagnostics to each episode for shared reporting."""
    if not batch.episodes:
        raise ValueError("PPO training batches must contain at least one episode")
    actor_loss = float(
        np.mean([update.actor_loss for update in batch.minibatch_updates])
    )
    critic_loss = float(
        np.mean([update.critic_loss for update in batch.minibatch_updates])
    )
    entropy = float(np.mean([update.entropy for update in batch.minibatch_updates]))
    updates_per_episode = len(batch.minibatch_updates) / len(batch.episodes)

    return tuple(
        TrainingEpisodeResult(
            episode_return=episode.episode_return,
            episode_length=episode.episode_length,
            terminated=episode.terminated,
            truncated=episode.truncated,
            updates=updates_per_episode,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            policy_entropy=entropy if collect_diagnostics else None,
        )
        for episode in batch.episodes
    )


def diagnostic_row(
    algorithm: str,
    seed: int,
    training_episode: int,
    actor_learning_rate: float,
    critic_learning_rate: float | None,
    result: TrainingEpisodeResult,
) -> dict[str, object]:
    """Convert one training episode to the optional diagnostic schema."""
    if result.policy_entropy is None:
        raise ValueError("training result does not contain diagnostics")
    return {
        "algorithm": algorithm,
        "seed": seed,
        "training_episode": training_episode,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": (
            "" if critic_learning_rate is None else critic_learning_rate
        ),
        "episode_return": result.episode_return,
        "episode_length": result.episode_length,
        "terminated": int(result.terminated),
        "truncated": int(result.truncated),
        "updates": result.updates,
        "actor_loss": result.actor_loss,
        "critic_loss": "" if result.critic_loss is None else result.critic_loss,
        "policy_entropy": result.policy_entropy,
    }


def print_diagnostic_summary(rows: list[dict[str, object]]) -> None:
    """Print compact diagnostic means for one completed checkpoint segment."""
    if not rows:
        return

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    critic_losses = [
        float(row["critic_loss"]) for row in rows if row["critic_loss"] != ""
    ]
    critic_text = (
        f" | critic loss={float(np.mean(critic_losses)):.3g}" if critic_losses else ""
    )
    print(
        "    DIAG  | "
        f"actor loss={mean('actor_loss'):.3g}{critic_text} | "
        f"entropy={mean('policy_entropy'):.3g} | "
        f"updates/episode={mean('updates'):.2f}",
        flush=True,
    )


def print_experiment_header(config: ExperimentConfig) -> None:
    """Print the resolved settings most useful while watching a run."""
    optimizer_name = {
        "adam": "Adam",
        "adamw": "AdamW",
        "sgd": "SGD",
    }[config.optimizer]
    warmup = (
        f"{config.warmup_episodes} episodes from {config.warmup_start_factor:g}x peak"
        if config.warmup_episodes
        else "disabled"
    )

    print("", flush=True)
    print("POLICY-GRADIENT EXPERIMENT", flush=True)
    print(f"  Environment : {config.environment}", flush=True)
    print(f"  Preset      : {config.preset}", flush=True)
    print(
        f"  Training    : {config.training_episodes} episodes per seed",
        flush=True,
    )
    print(
        f"  Evaluation  : {config.evaluation_episodes} episodes per checkpoint",
        flush=True,
    )
    print(
        f"  Seed trials : {config.seeds}; values={list(config.seed_values)}; "
        "fresh network each trial",
        flush=True,
    )
    print(
        f"  Algorithm   : {ALGORITHM_LABELS[config.algorithm]}",
        flush=True,
    )
    print(
        f"  Optimizer   : {optimizer_name}; weight decay={config.weight_decay:g}",
        flush=True,
    )
    print(
        f"  Actor LR    : {config.actor_learning_rate:g} -> "
        f"{config.actor_minimum_learning_rate:g}",
        flush=True,
    )
    print(
        f"  Critic LR   : {config.critic_learning_rate:g} -> "
        f"{config.critic_minimum_learning_rate:g}",
        flush=True,
    )
    print(f"  Warmup      : {warmup}", flush=True)
    if config.algorithm == "ppo":
        print(
            f"  PPO batch    : {config.ppo_batch_episodes} complete episodes; "
            f"{config.ppo_update_epochs} epochs; "
            f"minibatch={config.ppo_minibatch_size}",
            flush=True,
        )
        print(
            f"  PPO objective: clip={config.ppo_clip_ratio:g}; "
            f"GAE lambda={config.gae_lambda:g}; "
            f"entropy coefficient={config.entropy_coefficient:g}",
            flush=True,
        )
    else:
        print(
            f"  Actor-critic: rollout={config.rollout_steps} steps; "
            f"entropy coefficient={config.entropy_coefficient:g}; "
            f"A3C workers={config.a3c_workers}",
            flush=True,
        )
    print(
        "  Diagnostics : "
        + ("training_diagnostics.csv" if config.diagnostics else "off"),
        flush=True,
    )


def print_final_evaluation(
    rows: list[dict[str, object]],
    algorithms: Sequence[str],
) -> None:
    """Print final per-seed measurements and their across-seed aggregate."""
    final_rows = [row for row in rows if int(row["checkpoint"]) == 100]
    summaries: list[tuple[str, float, float, float, float]] = []
    print("", flush=True)
    print("FINAL FROZEN-POLICY EVALUATION", flush=True)
    print(
        f"  {'Algorithm':<23} {'Seed':>4} {'Mean return':>12} {'Success':>9}",
        flush=True,
    )
    print(f"  {'-' * 52}", flush=True)

    for algorithm in algorithms:
        algorithm_rows = [row for row in final_rows if row["algorithm"] == algorithm]
        seed_returns: list[float] = []
        seed_successes: list[float] = []
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = [row for row in algorithm_rows if int(row["seed"]) == seed]
            mean_return = float(
                np.mean([float(row["episode_return"]) for row in seed_rows])
            )
            success_rate = float(np.mean([float(row["success"]) for row in seed_rows]))
            seed_returns.append(mean_return)
            seed_successes.append(success_rate)
            print(
                f"  {ALGORITHM_LABELS[algorithm]:<23} {seed:>4} "
                f"{mean_return:>12.1f} {success_rate:>8.0%}",
                flush=True,
            )

        if not seed_returns:
            continue
        return_std = (
            float(np.std(seed_returns, ddof=1)) if len(seed_returns) > 1 else 0.0
        )
        success_std = (
            float(np.std(seed_successes, ddof=1)) if len(seed_successes) > 1 else 0.0
        )
        summaries.append(
            (
                ALGORITHM_LABELS[algorithm],
                float(np.mean(seed_returns)),
                return_std,
                float(np.mean(seed_successes)),
                success_std,
            )
        )

    print("", flush=True)
    print("ACROSS-SEED SUMMARY", flush=True)
    print(
        f"  {'Algorithm':<23} {'Return mean +/- SD':>20} {'Success mean +/- SD':>21}",
        flush=True,
    )
    print(f"  {'-' * 66}", flush=True)
    for label, return_mean, return_std, success_mean, success_std in summaries:
        print(
            f"  {label:<23} {return_mean:>8.1f} +/- {return_std:<7.1f} "
            f"{success_mean:>8.0%} +/- {success_std:.0%}",
            flush=True,
        )
    print("", flush=True)


def run_policy_gradient_experiment(
    config: ExperimentConfig,
    output: Path | None,
) -> tuple[list[dict[str, object]], int | None]:
    """Train and evaluate the selected algorithm with reproducible seeds."""
    rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    metrics_path = output / "metrics.csv" if output is not None else None
    diagnostics_path = (
        output / "training_diagnostics.csv"
        if config.diagnostics and output is not None
        else None
    )
    if diagnostics_path is not None:
        write_csv(diagnostics_path, DIAGNOSTIC_FIELDS, diagnostic_rows)
    algorithm = config.algorithm
    print("", flush=True)
    print("=" * 72, flush=True)
    print(f"ALGORITHM: {ALGORITHM_LABELS[algorithm]}", flush=True)
    print("=" * 72, flush=True)
    recording_states: dict[int, dict[int, dict[str, ModelState]]] = {}
    final_scores: dict[int, tuple[float, float]] = {}
    recording_agent: Agent | None = None
    checkpoints = evaluation_checkpoints(config.preset)
    for trial_index, seed in enumerate(config.seed_values):
        print("", flush=True)
        print(
            f"SEED {trial_index + 1}/{config.seeds} | seed={seed} | fresh network",
            flush=True,
        )
        print("-" * 72, flush=True)
        training_env = make_environment(config.environment)
        evaluation_env = make_environment(config.environment)
        agent = make_agent(
            algorithm,
            training_env,
            actor_learning_rate=config.actor_learning_rate,
            critic_learning_rate=config.critic_learning_rate,
            optimizer_name=config.optimizer,
            weight_decay=config.weight_decay,
            discount=config.discount,
            entropy_coefficient=config.entropy_coefficient,
            hidden_sizes=config.hidden_sizes,
            seed=seed,
            ppo_clip_ratio=config.ppo_clip_ratio,
        )
        agent_critic_optimizer = critic_optimizer(agent)
        actor_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        critic_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if algorithm != "a3c":
            actor_scheduler = make_learning_rate_scheduler(
                agent.actor_optimizer,
                training_episodes=config.training_episodes,
                minimum_learning_rate=config.actor_minimum_learning_rate,
                warmup_episodes=config.warmup_episodes,
                warmup_start_factor=config.warmup_start_factor,
            )
        if agent_critic_optimizer is not None and algorithm != "a3c":
            critic_scheduler = make_learning_rate_scheduler(
                agent_critic_optimizer,
                training_episodes=config.training_episodes,
                minimum_learning_rate=config.critic_minimum_learning_rate,
                warmup_episodes=config.warmup_episodes,
                warmup_start_factor=config.warmup_start_factor,
            )
        completed_episodes = 0
        try:
            for checkpoint in checkpoints:
                target_episodes = checkpoint_episode_target(
                    config.training_episodes,
                    checkpoint,
                )
                starting_episode = completed_episodes
                if target_episodes > starting_episode:
                    print(
                        f"  CHECKPOINT {checkpoint:>3}% | train episodes "
                        f"{starting_episode + 1}-{target_episodes}",
                        flush=True,
                    )
                else:
                    print(
                        f"  CHECKPOINT {checkpoint:>3}% | "
                        + (
                            "evaluate initial policy before training"
                            if checkpoint == 0
                            else "evaluate current policy; no additional training"
                        ),
                        flush=True,
                    )
                training_results: list[TrainingEpisodeResult] = []
                diagnostic_segment_start = len(diagnostic_rows)
                reported_actor_learning_rate = optimizer_learning_rate(
                    agent.actor_optimizer
                )
                reported_critic_learning_rate = (
                    optimizer_learning_rate(agent_critic_optimizer)
                    if agent_critic_optimizer is not None
                    else None
                )
                if algorithm == "a3c" and target_episodes > completed_episodes:
                    assert isinstance(agent, A2C)
                    assert agent_critic_optimizer is not None
                    max_episode_steps = (
                        training_env.spec.max_episode_steps
                        if training_env.spec is not None
                        else None
                    )
                    a3c_results = train_a3c_episodes(
                        config.environment,
                        agent.actor_model,
                        agent.actor_optimizer,
                        agent.critic_model,
                        agent.critic_optimizer,
                        episode_start=completed_episodes,
                        episode_count=target_episodes - completed_episodes,
                        total_episodes=config.training_episodes,
                        workers=config.a3c_workers,
                        rollout_steps=config.rollout_steps,
                        actor_learning_rate=config.actor_learning_rate,
                        actor_minimum_learning_rate=(
                            config.actor_minimum_learning_rate
                        ),
                        critic_learning_rate=config.critic_learning_rate,
                        critic_minimum_learning_rate=(
                            config.critic_minimum_learning_rate
                        ),
                        warmup_episodes=config.warmup_episodes,
                        warmup_start_factor=config.warmup_start_factor,
                        discount=config.discount,
                        entropy_coefficient=config.entropy_coefficient,
                        seed=seed,
                        collect_diagnostics=config.diagnostics,
                        max_episode_steps=max_episode_steps,
                    )
                    training_results.extend(item.result for item in a3c_results)
                    if a3c_results:
                        reported_actor_learning_rate = a3c_results[
                            -1
                        ].actor_learning_rate
                        reported_critic_learning_rate = a3c_results[
                            -1
                        ].critic_learning_rate
                    if diagnostics_path is not None:
                        for item in a3c_results:
                            diagnostic_rows.append(
                                diagnostic_row(
                                    algorithm,
                                    seed,
                                    item.episode_index + 1,
                                    item.actor_learning_rate,
                                    item.critic_learning_rate,
                                    item.result,
                                )
                            )
                    completed_episodes = target_episodes
                elif algorithm == "ppo" and target_episodes > completed_episodes:
                    assert isinstance(agent, PPO)
                    assert agent_critic_optimizer is not None
                    assert actor_scheduler is not None
                    while completed_episodes < target_episodes:
                        batch_episode_count = min(
                            config.ppo_batch_episodes,
                            target_episodes - completed_episodes,
                        )
                        episode_indices = tuple(
                            range(
                                completed_episodes,
                                completed_episodes + batch_episode_count,
                            )
                        )
                        actor_learning_rate = optimizer_learning_rate(
                            agent.actor_optimizer
                        )
                        critic_learning_rate = optimizer_learning_rate(
                            agent_critic_optimizer
                        )
                        batch = train_ppo_episodes(
                            training_env,
                            agent,
                            environment_seeds=tuple(
                                seed * config.training_episodes + episode_index
                                for episode_index in episode_indices
                            ),
                            action_seeds=tuple(
                                10_000_000
                                + seed * config.training_episodes
                                + episode_index
                                for episode_index in episode_indices
                            ),
                            discount=config.discount,
                            gae_lambda=config.gae_lambda,
                            update_epochs=config.ppo_update_epochs,
                            minibatch_size=config.ppo_minibatch_size,
                        )
                        batch_training_results = ppo_episode_training_results(
                            batch,
                            collect_diagnostics=config.diagnostics,
                        )
                        training_results.extend(batch_training_results)
                        if diagnostics_path is not None:
                            for episode_index, result in zip(
                                episode_indices,
                                batch_training_results,
                                strict=True,
                            ):
                                diagnostic_rows.append(
                                    diagnostic_row(
                                        algorithm,
                                        seed,
                                        episode_index + 1,
                                        actor_learning_rate,
                                        critic_learning_rate,
                                        result,
                                    )
                                )
                        for _ in episode_indices:
                            actor_scheduler.step()
                            if critic_scheduler is not None:
                                critic_scheduler.step()
                        completed_episodes += batch_episode_count

                    reported_actor_learning_rate = optimizer_learning_rate(
                        agent.actor_optimizer
                    )
                    reported_critic_learning_rate = optimizer_learning_rate(
                        agent_critic_optimizer
                    )
                else:
                    while completed_episodes < target_episodes:
                        episode_index = completed_episodes
                        actor_learning_rate = optimizer_learning_rate(
                            agent.actor_optimizer
                        )
                        critic_learning_rate = (
                            optimizer_learning_rate(agent_critic_optimizer)
                            if agent_critic_optimizer is not None
                            else None
                        )
                        result = train_episode(
                            training_env,
                            agent,
                            rollout_steps=config.rollout_steps,
                            environment_seed=(
                                seed * config.training_episodes + episode_index
                            ),
                            action_seed=(
                                10_000_000
                                + seed * config.training_episodes
                                + episode_index
                            ),
                            collect_diagnostics=config.diagnostics,
                        )
                        training_results.append(result)
                        if diagnostics_path is not None:
                            diagnostic_rows.append(
                                diagnostic_row(
                                    algorithm,
                                    seed,
                                    episode_index + 1,
                                    actor_learning_rate,
                                    critic_learning_rate,
                                    result,
                                )
                            )
                        assert actor_scheduler is not None
                        actor_scheduler.step()
                        if critic_scheduler is not None:
                            critic_scheduler.step()
                        completed_episodes += 1
                    reported_actor_learning_rate = optimizer_learning_rate(
                        agent.actor_optimizer
                    )
                    reported_critic_learning_rate = (
                        optimizer_learning_rate(agent_critic_optimizer)
                        if agent_critic_optimizer is not None
                        else None
                    )

                if training_results:
                    mean_training_return = float(
                        np.mean([r.episode_return for r in training_results])
                    )
                    mean_training_length = float(
                        np.mean([r.episode_length for r in training_results])
                    )
                    mean_updates = float(np.mean([r.updates for r in training_results]))
                    print(
                        "    TRAIN | "
                        f"mean return={mean_training_return:.1f} | "
                        f"mean length={mean_training_length:.1f} | "
                        f"updates/episode={mean_updates:.2f} | "
                        "actor lr="
                        f"{reported_actor_learning_rate:.3g}"
                        + (
                            f" | critic lr={reported_critic_learning_rate:.3g}"
                            if reported_critic_learning_rate is not None
                            else ""
                        ),
                        flush=True,
                    )
                if diagnostics_path is not None:
                    diagnostic_segment = diagnostic_rows[diagnostic_segment_start:]
                    print_diagnostic_summary(diagnostic_segment)
                    write_csv(
                        diagnostics_path,
                        DIAGNOSTIC_FIELDS,
                        diagnostic_rows,
                    )

                checkpoint_results: list[EvaluationResult] = []
                for evaluation_episode in range(config.evaluation_episodes):
                    environment_seed = (
                        1_000_000
                        + seed * 100_000
                        + checkpoint * 1_000
                        + evaluation_episode
                    )
                    action_seed = (
                        11_000_000
                        + seed * 100_000
                        + checkpoint * 1_000
                        + evaluation_episode
                    )
                    result = evaluate_episode(
                        evaluation_env,
                        config.environment,
                        agent,
                        environment_seed=environment_seed,
                        action_seed=action_seed,
                    )
                    checkpoint_results.append(result)
                    rows.append(
                        evaluation_row(
                            algorithm,
                            seed,
                            checkpoint,
                            evaluation_episode,
                            result,
                        )
                    )

                mean_return = float(
                    np.mean([result.episode_return for result in checkpoint_results])
                )
                success_rate = float(
                    np.mean([result.success for result in checkpoint_results])
                )
                print(
                    f"    EVAL  | mean return={mean_return:.1f} | "
                    f"success={success_rate:.0%} | "
                    f"episodes={config.evaluation_episodes}",
                    flush=True,
                )
                if metrics_path is not None:
                    write_csv(metrics_path, CSV_FIELDS, rows)

                if checkpoint in config.recording_checkpoints or (
                    output is not None
                    and config.preset == "standard"
                    and checkpoint == 100
                ):
                    snapshot = {
                        "actor": {
                            name: value.detach().clone()
                            for name, value in (agent.actor_model.state_dict().items())
                        }
                    }
                    agent_critic_model = getattr(agent, "critic_model", None)
                    if agent_critic_model is not None:
                        snapshot["critic"] = {
                            name: value.detach().clone()
                            for name, value in (agent_critic_model.state_dict().items())
                        }
                    recording_states.setdefault(seed, {})[checkpoint] = snapshot
                if checkpoint == 100:
                    final_scores[seed] = (mean_return, success_rate)
        finally:
            training_env.close()
            evaluation_env.close()
        recording_agent = agent

    if config.recording_checkpoints:
        selected_seed = max(final_scores, key=final_scores.__getitem__)
        assert recording_agent is not None
        assert output is not None
        selected_snapshot = recording_states[selected_seed][100]
        if config.preset == "standard":
            torch.save(
                {
                    "algorithm": algorithm,
                    "seed": selected_seed,
                    "checkpoint": 100,
                    "actor_state_dict": selected_snapshot["actor"],
                    "critic_state_dict": selected_snapshot.get("critic"),
                },
                output / "best_model.pt",
            )
        print(
            f"  RECORDINGS | selected seed={selected_seed} from final evaluation",
            flush=True,
        )
        for checkpoint in config.recording_checkpoints:
            recording_agent.actor_model.load_state_dict(
                recording_states[selected_seed][checkpoint]["actor"]
            )
            record_evaluation(
                output / "recordings" / f"{algorithm}_checkpoint_{checkpoint:03d}.gif",
                config.environment,
                recording_agent,
                environment_seed=2_000_000 + checkpoint,
                action_seed=12_000_000 + checkpoint,
                frame_stride=recording_frame_stride(config.environment),
            )
            print(f"    recorded checkpoint {checkpoint}%", flush=True)
        return rows, selected_seed
    return rows, None


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_config(argv)
    output = (
        None
        if config.preset == "quick"
        else create_run_directory(
            "policy_gradient",
            config.environment,
            config.preset,
        )
    )
    if output is not None:
        (output / "figures").mkdir()
        if config.recording_checkpoints:
            (output / "recordings").mkdir()

    metadata = initial_metadata(config)
    if output is not None:
        write_metadata(output / "metadata.json", metadata)
        write_csv(output / "metrics.csv", CSV_FIELDS, [])
    print_experiment_header(config)
    try:
        rows, selected_seed = run_policy_gradient_experiment(config, output)
        if output is None:
            print_final_evaluation(rows, (config.algorithm,))
            print(
                "Quick compatibility run complete; no artifacts were written.",
                flush=True,
            )
            return
        metadata["recording_seed_selection"] = (
            "highest final mean evaluation return; success rate breaks ties"
            if selected_seed is not None
            else None
        )
        metadata["selected_seed"] = selected_seed
        metadata["model_file"] = (
            "best_model.pt" if config.preset == "standard" else None
        )
        metadata["model_seed"] = selected_seed if config.preset == "standard" else None
        write_report(output, config, rows, metadata)
    except Exception:
        if output is not None:
            metadata["status"] = "failed"
            write_metadata(output / "metadata.json", metadata)
        raise

    metadata["status"] = "complete"
    write_metadata(output / "metadata.json", metadata)
    print_final_evaluation(rows, (config.algorithm,))
    print("EXPERIMENT COMPLETE", flush=True)
    print(f"  Output  : {output}", flush=True)
    print(f"  Summary : {output / 'summary.html'}", flush=True)
    print(f"  Metrics : {output / 'metrics.csv'}", flush=True)
    if config.diagnostics:
        print(f"  Diag    : {output / 'training_diagnostics.csv'}", flush=True)


if __name__ == "__main__":
    main()
