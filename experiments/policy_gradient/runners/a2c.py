"""A2C experiment training backend."""

import math

import gymnasium as gym
import torch

from experiments.policy_gradient.runners.common import (
    Observation,
    TrainingEpisodeResult,
    mean_policy_entropy,
    observation_array,
)
from rl_lib.algorithms.policy_gradient import A2C
from rl_lib.data import EpisodeStep


def train_episode(
    env: gym.Env,
    agent: A2C,
    *,
    rollout_steps: int,
    environment_seed: int,
    action_seed: int,
    collect_diagnostics: bool,
) -> TrainingEpisodeResult:
    """Train A2C during one episode using bounded rollouts."""
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
                raise RuntimeError("A2C produced a non-finite loss")
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
