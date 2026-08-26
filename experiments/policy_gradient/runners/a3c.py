"""Multiprocessing training backend for A3C experiments."""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from queue import Empty
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp

from experiments.policy_gradient.environments import make_environment
from experiments.policy_gradient.runners.common import (
    Observation,
    TrainingEpisodeResult,
    mean_policy_entropy,
    observation_array,
)
from rl_lib.algorithms.policy_gradient import A3C
from rl_lib.data import EpisodeStep
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork
from rl_lib.optimizers import share_optimizer_state


@dataclass(frozen=True)
class A3CEpisodeResult:
    episode_index: int
    worker: int
    actor_learning_rate: float
    critic_learning_rate: float
    result: TrainingEpisodeResult


@dataclass(frozen=True)
class _WorkerConfig:
    environment: str
    max_episode_steps: int | None
    episode_end: int
    total_episodes: int
    rollout_steps: int
    actor_learning_rate: float
    actor_minimum_learning_rate: float
    critic_learning_rate: float
    critic_minimum_learning_rate: float
    warmup_episodes: int
    warmup_start_factor: float
    discount: float
    entropy_coefficient: float
    seed: int
    collect_diagnostics: bool


@dataclass(frozen=True)
class _WorkerFailure:
    worker: int
    message: str
    traceback: str


def scheduled_learning_rate(
    episode_index: int,
    *,
    total_episodes: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
    warmup_episodes: int,
    warmup_start_factor: float,
) -> float:
    """Return the warmup-cosine rate used by one zero-based episode."""
    minimum_factor = minimum_learning_rate / peak_learning_rate
    if warmup_episodes and episode_index <= warmup_episodes:
        progress = episode_index / warmup_episodes
        factor = warmup_start_factor + progress * (1.0 - warmup_start_factor)
    else:
        cosine_episodes = max(1, total_episodes - warmup_episodes)
        progress = (episode_index - warmup_episodes) / cosine_episodes
        factor = minimum_factor + 0.5 * (1.0 - minimum_factor) * (
            1.0 + math.cos(math.pi * progress)
        )
    return peak_learning_rate * factor


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _train_episode(
    env: gym.Env,
    agent: A3C,
    *,
    rollout_steps: int,
    environment_seed: int,
    action_seed: int,
    collect_diagnostics: bool,
) -> TrainingEpisodeResult:
    observation, _ = env.reset(seed=environment_seed)
    state = observation_array(observation, agent.actor_model.observation_size)
    rollout: list[EpisodeStep[Observation]] = []
    terminated = truncated = False
    episode_return = 0.0
    episode_length = 0
    updates = 0
    weighted_actor_loss = 0.0
    weighted_critic_loss = 0.0
    weighted_entropy = 0.0

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(action_seed)
        while not (terminated or truncated):
            action = agent.select_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_array(
                next_observation,
                agent.actor_model.observation_size,
            )
            rollout.append(EpisodeStep(state, action, float(reward)))
            episode_return += float(reward)
            episode_length += 1
            state = next_state

            if len(rollout) < rollout_steps and not (terminated or truncated):
                continue

            rollout_size = len(rollout)
            if collect_diagnostics:
                weighted_entropy += mean_policy_entropy(agent, rollout) * rollout_size
            actor_loss, critic_loss = agent.update(
                tuple(rollout),
                state,
                terminated=terminated,
            )
            if not (math.isfinite(actor_loss) and math.isfinite(critic_loss)):
                raise RuntimeError("A3C produced a non-finite loss")
            weighted_actor_loss += actor_loss * rollout_size
            weighted_critic_loss += critic_loss * rollout_size
            updates += 1
            rollout.clear()

    return TrainingEpisodeResult(
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=terminated,
        truncated=truncated,
        updates=updates,
        actor_loss=weighted_actor_loss / episode_length,
        critic_loss=weighted_critic_loss / episode_length,
        policy_entropy=(
            weighted_entropy / episode_length if collect_diagnostics else None
        ),
    )


def _worker(
    worker: int,
    config: _WorkerConfig,
    shared_actor_model: DiscretePolicyNetwork,
    shared_actor_optimizer: torch.optim.Optimizer,
    shared_critic_model: StateValueNetwork,
    shared_critic_optimizer: torch.optim.Optimizer,
    update_lock: Any,
    next_episode: Any,
    result_queue: Any,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(100_000 + config.seed * 1_000 + worker)
    np.random.seed(200_000 + config.seed * 1_000 + worker)
    env: gym.Env | None = None
    try:
        env = make_environment(
            config.environment,
            max_episode_steps=config.max_episode_steps,
        )
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("A3C environment must have a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("A3C environment must have a Discrete action space")

        actor_model = DiscretePolicyNetwork(
            shared_actor_model.observation_size,
            shared_actor_model.number_of_actions,
            shared_actor_model.hidden_sizes,
        )
        critic_model = StateValueNetwork(
            shared_critic_model.observation_size,
            shared_critic_model.hidden_sizes,
        )
        agent = A3C(
            actor_model,
            shared_actor_model,
            shared_actor_optimizer,
            critic_model,
            shared_critic_model,
            shared_critic_optimizer,
            update_lock,
            config.discount,
            config.entropy_coefficient,
        )

        while True:
            with next_episode.get_lock():
                if next_episode.value >= config.episode_end:
                    break
                episode_index = int(next_episode.value)
                next_episode.value += 1

            actor_learning_rate = scheduled_learning_rate(
                episode_index,
                total_episodes=config.total_episodes,
                peak_learning_rate=config.actor_learning_rate,
                minimum_learning_rate=config.actor_minimum_learning_rate,
                warmup_episodes=config.warmup_episodes,
                warmup_start_factor=config.warmup_start_factor,
            )
            critic_learning_rate = scheduled_learning_rate(
                episode_index,
                total_episodes=config.total_episodes,
                peak_learning_rate=config.critic_learning_rate,
                minimum_learning_rate=config.critic_minimum_learning_rate,
                warmup_episodes=config.warmup_episodes,
                warmup_start_factor=config.warmup_start_factor,
            )
            _set_learning_rate(shared_actor_optimizer, actor_learning_rate)
            _set_learning_rate(shared_critic_optimizer, critic_learning_rate)
            result = _train_episode(
                env,
                agent,
                rollout_steps=config.rollout_steps,
                environment_seed=config.seed * config.total_episodes + episode_index,
                action_seed=(
                    10_000_000 + config.seed * config.total_episodes + episode_index
                ),
                collect_diagnostics=config.collect_diagnostics,
            )
            result_queue.put(
                A3CEpisodeResult(
                    episode_index,
                    worker,
                    actor_learning_rate,
                    critic_learning_rate,
                    result,
                )
            )
    except Exception as error:
        result_queue.put(_WorkerFailure(worker, str(error), traceback.format_exc()))
    finally:
        if env is not None:
            env.close()


def _validate_training_request(
    *,
    episode_start: int,
    episode_count: int,
    total_episodes: int,
    workers: int,
    rollout_steps: int,
    warmup_episodes: int,
) -> None:
    if episode_start < 0 or episode_count < 0:
        raise ValueError("Episode start and count must be nonnegative")
    if total_episodes < 1 or episode_start + episode_count > total_episodes:
        raise ValueError("Episode range must stay inside the training budget")
    if workers < 1:
        raise ValueError("A3C workers must be positive")
    if rollout_steps < 1:
        raise ValueError("A3C rollout steps must be positive")
    if not 0 <= warmup_episodes < total_episodes:
        raise ValueError("Warmup episodes must stay inside the training budget")


def train_episodes(
    environment: str,
    shared_actor_model: DiscretePolicyNetwork,
    shared_actor_optimizer: torch.optim.Optimizer,
    shared_critic_model: StateValueNetwork,
    shared_critic_optimizer: torch.optim.Optimizer,
    *,
    episode_start: int,
    episode_count: int,
    total_episodes: int,
    workers: int,
    rollout_steps: int,
    actor_learning_rate: float,
    actor_minimum_learning_rate: float,
    critic_learning_rate: float,
    critic_minimum_learning_rate: float,
    warmup_episodes: int,
    warmup_start_factor: float,
    discount: float,
    entropy_coefficient: float,
    seed: int,
    collect_diagnostics: bool,
    max_episode_steps: int | None = None,
) -> list[A3CEpisodeResult]:
    """Train one exact episode range with asynchronously updating workers."""
    _validate_training_request(
        episode_start=episode_start,
        episode_count=episode_count,
        total_episodes=total_episodes,
        workers=workers,
        rollout_steps=rollout_steps,
        warmup_episodes=warmup_episodes,
    )
    if episode_count == 0:
        return []

    shared_actor_model.share_memory()
    shared_critic_model.share_memory()
    share_optimizer_state(shared_actor_optimizer)
    share_optimizer_state(shared_critic_optimizer)

    config = _WorkerConfig(
        environment,
        max_episode_steps,
        episode_start + episode_count,
        total_episodes,
        rollout_steps,
        actor_learning_rate,
        actor_minimum_learning_rate,
        critic_learning_rate,
        critic_minimum_learning_rate,
        warmup_episodes,
        warmup_start_factor,
        discount,
        entropy_coefficient,
        seed,
        collect_diagnostics,
    )
    context = mp.get_context("spawn")
    update_lock = context.Lock()
    next_episode = context.Value("q", episode_start)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            name=f"a3c-worker-{worker}",
            args=(
                worker,
                config,
                shared_actor_model,
                shared_actor_optimizer,
                shared_critic_model,
                shared_critic_optimizer,
                update_lock,
                next_episode,
                result_queue,
            ),
        )
        for worker in range(min(workers, episode_count))
    ]

    results: list[A3CEpisodeResult] = []
    try:
        for process in processes:
            process.start()

        while len(results) < episode_count:
            try:
                message = result_queue.get(timeout=1.0)
            except Empty:
                failed = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(
                        f"A3C worker exited with code {failed[0].exitcode}"
                    ) from None
                if all(process.exitcode == 0 for process in processes):
                    raise RuntimeError(
                        "A3C workers exited before training completed"
                    ) from None
                continue

            if isinstance(message, _WorkerFailure):
                raise RuntimeError(
                    f"A3C worker {message.worker} failed: {message.message}\n"
                    f"{message.traceback}"
                )
            if not isinstance(message, A3CEpisodeResult):
                raise RuntimeError("A3C worker returned an unknown message")
            results.append(message)

        for process in processes:
            process.join(timeout=10)
        if any(process.is_alive() for process in processes):
            raise RuntimeError("A3C worker did not stop after training")
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"A3C worker exited with code {failed[0].exitcode}")
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        raise
    finally:
        result_queue.close()
        result_queue.join_thread()
        for process in processes:
            process.close()

    return sorted(results, key=lambda item: item.episode_index)
