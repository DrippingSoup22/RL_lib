import math
from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from experiments.policy_gradient.runners.common import Observation, observation_array
from rl_lib.algorithms.policy_gradient import (
    PPO,
    generalized_advantage_estimates,
)
from rl_lib.data import ContinuousPPOActionSample, PPOUpdateResult


@dataclass(frozen=True)
class PPOEpisodeResult:
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class PPOTrainingBatchResult:
    episodes: tuple[PPOEpisodeResult, ...]
    minibatch_updates: tuple[PPOUpdateResult, ...]


def train_episodes(
    env: gym.Env,
    agent: PPO,
    *,
    environment_seeds: Sequence[int],
    action_seeds: Sequence[int],
    discount: float,
    gae_lambda: float,
    update_epochs: int,
    minibatch_size: int,
    reward_scale: float = 1.0,
) -> PPOTrainingBatchResult:
    """Collect complete episodes, calculate GAE, then update one frozen batch."""

    if len(environment_seeds) == 0 or len(action_seeds) == 0:
        raise ValueError("Environment and action seeds must be nonempty")
    if len(environment_seeds) != len(action_seeds):
        raise ValueError("Environment and action seeds must have the same length")

    batch_observations: list[Observation] = []
    batch_policy_actions: list[int | NDArray[np.float32]] = []
    batch_old_log_probabilities: list[float] = []
    batch_advantages: list[float] = []
    batch_return_targets: list[float] = []
    episode_results: list[PPOEpisodeResult] = []

    for environment_seed, action_seed in zip(
        environment_seeds,
        action_seeds,
        strict=True,
    ):
        observation, _ = env.reset(seed=environment_seed)
        state = observation_array(observation, agent.actor_model.observation_size)
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0
        episode_rewards: list[float] = []
        episode_values: list[float] = []

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_seed)
            while not (terminated or truncated):
                sample = agent.sample_action(state)
                next_observation, reward, terminated, truncated, _ = env.step(
                    sample.action
                )

                batch_observations.append(state)
                # Categorical actions are reevaluated directly. Continuous PPO
                # instead stores the raw Gaussian sample, while only its bounded
                # transformation is sent to the environment above.
                batch_policy_actions.append(
                    sample.latent_action
                    if isinstance(sample, ContinuousPPOActionSample)
                    else sample.action
                )
                batch_old_log_probabilities.append(sample.log_probability)
                episode_rewards.append(float(reward))
                episode_values.append(sample.value)

                state = observation_array(
                    next_observation,
                    agent.actor_model.observation_size,
                )
                episode_length += 1
                episode_return += float(reward) / reward_scale

        # A true terminal state has no future value. A truncation still does.
        final_value = 0.0 if terminated else agent.state_value(state)

        advantages, return_targets = generalized_advantage_estimates(
            episode_rewards,
            episode_values,
            final_value,
            terminated=terminated,
            discount=discount,
            gae_lambda=gae_lambda,
        )

        # Keep these aligned transition-for-transition with the collected data.
        batch_advantages.extend(float(value) for value in advantages)
        batch_return_targets.extend(float(value) for value in return_targets)
        episode_results.append(
            PPOEpisodeResult(
                episode_return=episode_return,
                episode_length=episode_length,
                terminated=terminated,
                truncated=truncated,
            )
        )

    minibatch_updates = agent.update(
        np.asarray(batch_observations, dtype=np.float32),
        np.asarray(batch_policy_actions),
        np.asarray(batch_old_log_probabilities, dtype=np.float32),
        np.asarray(batch_advantages, dtype=np.float32),
        np.asarray(batch_return_targets, dtype=np.float32),
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
    )
    if any(
        not all(
            math.isfinite(value)
            for value in (
                update.actor_loss,
                update.critic_loss,
                update.entropy,
                update.approximate_kl,
                update.clip_fraction,
            )
        )
        for update in minibatch_updates
    ):
        raise RuntimeError("PPO produced a non-finite update result")

    return PPOTrainingBatchResult(
        episodes=tuple(episode_results),
        minibatch_updates=minibatch_updates,
    )
