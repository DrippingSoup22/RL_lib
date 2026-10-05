import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from experiments.policy_gradient.runners.common import (
    Observation,
    environment_action,
    observation_array,
)
from rl_lib.algorithms.policy_gradient import (
    PPO,
    PPOUpdateSummary,
    generalized_advantage_estimates,
)


@dataclass(frozen=True)
class PPOEpisodeResult:
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class PPOTrainingBatchResult:
    episodes: tuple[PPOEpisodeResult, ...]
    update: PPOUpdateSummary
    minibatch_count: int


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
    """Collect complete episodes, calculate GAE, then update one frozen batch.

    PPO acts on batches of tensors; here each batch is the one observation of
    a single Gymnasium environment.
    """

    if len(environment_seeds) == 0 or len(action_seeds) == 0:
        raise ValueError("Environment and action seeds must be nonempty")
    if len(environment_seeds) != len(action_seeds):
        raise ValueError("Environment and action seeds must have the same length")

    batch_observations: list[Observation] = []
    batch_policy_actions: list[torch.Tensor] = []
    batch_old_log_probabilities: list[torch.Tensor] = []
    batch_advantages: list[torch.Tensor] = []
    batch_return_targets: list[torch.Tensor] = []
    episode_results: list[PPOEpisodeResult] = []

    def as_batch(state: Observation) -> torch.Tensor:
        return torch.as_tensor(state, device=agent.device).unsqueeze(0)

    for environment_seed, action_seed in zip(
        environment_seeds,
        action_seeds,
        strict=True,
    ):
        observation, _ = env.reset(seed=environment_seed)
        state = observation_array(observation, agent.actor_network.observation_size)
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0
        episode_rewards: list[float] = []
        episode_values: list[torch.Tensor] = []

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_seed)
            while not (terminated or truncated):
                sample = agent.sample_action(as_batch(state))
                next_observation, reward, terminated, truncated, _ = env.step(
                    environment_action(sample.environment_action[0])
                )

                # Learning re-evaluates the policy action: the action itself for
                # categorical PPO, the latent Gaussian sample for continuous PPO.
                batch_observations.append(state)
                batch_policy_actions.append(sample.policy_action[0])
                batch_old_log_probabilities.append(sample.log_probability[0])
                episode_rewards.append(float(reward))
                episode_values.append(sample.value[0])

                state = observation_array(
                    next_observation,
                    agent.actor_network.observation_size,
                )
                episode_length += 1
                episode_return += float(reward) / reward_scale

        # A true terminal state has no future value. A truncation still does.
        final_value = (
            torch.zeros((), device=agent.device)
            if terminated
            else agent.state_value(as_batch(state))[0]
        )

        # Inside the episode, a step's next value is the following step's own
        # value; after the final step it is the final state's. Collection runs
        # whole episodes, so the episode ends exactly on its final step.
        values = torch.stack(episode_values)
        next_values = torch.cat((values[1:], final_value.unsqueeze(0)))
        final_step = torch.arange(episode_length, device=agent.device) == (
            episode_length - 1
        )
        advantages, return_targets = generalized_advantage_estimates(
            torch.tensor(episode_rewards, device=agent.device),
            values,
            next_values,
            terminated=final_step & bool(terminated),
            episode_ended=final_step,
            discount=discount,
            gae_lambda=gae_lambda,
        )

        # Keep these aligned transition-for-transition with the collected data.
        batch_advantages.append(advantages)
        batch_return_targets.append(return_targets)
        episode_results.append(
            PPOEpisodeResult(
                episode_return=episode_return,
                episode_length=episode_length,
                terminated=terminated,
                truncated=truncated,
            )
        )

    sample_count = len(batch_observations)
    summary = agent.update(
        torch.as_tensor(np.asarray(batch_observations), device=agent.device),
        torch.stack(batch_policy_actions),
        torch.stack(batch_old_log_probabilities),
        torch.cat(batch_advantages),
        torch.cat(batch_return_targets),
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
    )
    if not all(math.isfinite(value) for value in dataclasses.astuple(summary)):
        raise RuntimeError("PPO produced a non-finite update result")

    return PPOTrainingBatchResult(
        episodes=tuple(episode_results),
        update=summary,
        minibatch_count=update_epochs * math.ceil(sample_count / minibatch_size),
    )
