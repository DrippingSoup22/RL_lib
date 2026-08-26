"""REINFORCE experiment training backend."""

import math

import gymnasium as gym

from experiments.policy_gradient.runners.common import (
    TrainingEpisodeResult,
    generate_episode,
    mean_policy_entropy,
)
from rl_lib.algorithms.policy_gradient import Reinforce, ReinforceWithBaseline

ReinforceAgent = Reinforce | ReinforceWithBaseline


def train_episode(
    env: gym.Env,
    agent: ReinforceAgent,
    *,
    environment_seed: int,
    action_seed: int,
    collect_diagnostics: bool,
) -> TrainingEpisodeResult:
    """Train one REINFORCE variant after sampling a complete episode."""
    episode = generate_episode(
        env,
        agent,
        environment_seed=environment_seed,
        action_seed=action_seed,
    )
    entropy = mean_policy_entropy(agent, episode.steps) if collect_diagnostics else None
    actor_loss = agent.update(episode)
    if not math.isfinite(actor_loss):
        raise RuntimeError("REINFORCE produced a non-finite loss")
    return TrainingEpisodeResult(
        episode_return=sum(step.reward for step in episode.steps),
        episode_length=len(episode.steps),
        terminated=episode.terminated,
        truncated=episode.truncated,
        updates=1,
        actor_loss=actor_loss,
        critic_loss=None,
        policy_entropy=entropy,
    )
