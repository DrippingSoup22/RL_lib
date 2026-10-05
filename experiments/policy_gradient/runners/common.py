"""Small data and interaction helpers shared by policy-gradient backends."""

from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from rl_lib.algorithms.policy_gradient import (
    A2C,
    A3C,
    PPO,
    Reinforce,
    ReinforceWithBaseline,
)
from rl_lib.trajectories import Episode, EpisodeStep

PolicyAgent = Reinforce | ReinforceWithBaseline | A2C | A3C | PPO
Observation = NDArray[np.float32]


@dataclass(frozen=True)
class TrainingEpisodeResult:
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    updates: float
    actor_loss: float
    critic_loss: float | None
    policy_entropy: float | None
    approximate_kl: float | None = None
    clip_fraction: float | None = None


def observation_array(observation: object, observation_size: int) -> Observation:
    """Convert a Gymnasium observation to one owned, flat float32 array."""
    result = np.asarray(observation, dtype=np.float32).reshape(-1)
    if result.shape != (observation_size,):
        raise ValueError("environment observation does not match the model input size")
    if not np.all(np.isfinite(result)):
        raise ValueError("environment observation must be finite")
    return result.copy()


def environment_action(action: torch.Tensor) -> int | NDArray[np.float32]:
    """One PPO action as Gymnasium expects it: an integer or a float32 array."""
    if action.ndim == 0:
        return int(action.item())
    return action.cpu().numpy().astype(np.float32)


def select_environment_action(
    agent: PolicyAgent,
    state: Observation,
    *,
    deterministic: bool,
) -> int | NDArray[np.float32]:
    """Choose an evaluation action without changing the agent.

    PPO acts on batches of tensors, so its batch here is the one observation.
    """
    if not isinstance(agent, PPO):
        return agent.select_action(state, deterministic=deterministic)
    observations = torch.as_tensor(state, device=agent.device).unsqueeze(0)
    return environment_action(
        agent.select_action(observations, deterministic=deterministic)[0]
    )


def generate_episode(
    env: gym.Env,
    agent: PolicyAgent,
    *,
    environment_seed: int,
    action_seed: int,
    deterministic: bool = False,
) -> Episode[Observation]:
    """Sample one complete episode without changing the agent."""
    observation, _ = env.reset(seed=environment_seed)
    state = observation_array(observation, agent.actor_network.observation_size)
    steps: list[EpisodeStep[Observation]] = []
    terminated = truncated = False

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(action_seed)
        while not (terminated or truncated):
            if deterministic or isinstance(agent, PPO):
                # PPO has a separate frozen-batch collector; this shared path is
                # used only for its evaluation episodes.
                action = select_environment_action(
                    agent, state, deterministic=deterministic
                )
                policy_action = None
            else:
                action, policy_action = agent.sample_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            steps.append(
                EpisodeStep(
                    state,
                    action,
                    float(reward),
                    policy_action=policy_action,
                )
            )
            state = observation_array(
                next_observation,
                agent.actor_network.observation_size,
            )

    return Episode(
        steps=tuple(steps),
        final_state=state,
        terminated=terminated,
        truncated=truncated,
    )


def mean_policy_entropy(
    agent: PolicyAgent,
    steps: Sequence[EpisodeStep[Observation]],
) -> float:
    """Measure the current policy's entropy on rollout observations."""
    observations = torch.as_tensor(
        np.asarray([step.state for step in steps], dtype=np.float32)
    )
    with torch.no_grad():
        entropy = agent.policy.entropy(observations).mean()
    return float(entropy.item())
