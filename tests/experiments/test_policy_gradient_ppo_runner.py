import math

import gymnasium as gym
import torch

from experiments.policy_gradient.environments import make_environment
from experiments.policy_gradient.runners.ppo import train_episodes
from rl_lib.algorithms.policy_gradient import PPO
from rl_lib.networks import (
    CategoricalPolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)


def test_ppo_runner_updates_one_batch_of_complete_cartpole_episodes() -> None:
    env = make_environment("CartPole-v1", max_episode_steps=3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        actor_network = CategoricalPolicyNetwork(4, 2, hidden_sizes=(4,))
        critic_network = StateValueNetwork(4, hidden_sizes=(4,))
    actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=0.001)
    agent = PPO(
        actor_network,
        actor_optimizer,
        critic_network,
        critic_optimizer,
        shuffle_seed=0,
    )
    actor_before = {
        name: value.detach().clone()
        for name, value in actor_network.state_dict().items()
    }

    result = train_episodes(
        env,
        agent,
        environment_seeds=(0, 1),
        action_seeds=(10, 11),
        discount=0.99,
        gae_lambda=0.95,
        update_epochs=2,
        minibatch_size=4,
    )
    env.close()

    assert len(result.episodes) == 2
    assert all(episode.episode_length == 3 for episode in result.episodes)
    assert all(episode.truncated for episode in result.episodes)
    assert all(not episode.terminated for episode in result.episodes)
    # Six samples in minibatches of four: two minibatches in each of two epochs.
    assert result.minibatch_count == 4
    assert all(
        math.isfinite(value)
        for value in (
            result.update.actor_loss,
            result.update.critic_loss,
            result.update.entropy,
        )
    )
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in actor_network.state_dict().items()
    )


def test_ppo_runner_updates_one_batch_of_continuous_pendulum_episodes() -> None:
    env = gym.make("Pendulum-v1", max_episode_steps=3)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    observation_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.shape[0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        actor_network = GaussianPolicyNetwork(
            observation_size,
            action_size,
            hidden_sizes=(4,),
        )
        critic_network = StateValueNetwork(observation_size, hidden_sizes=(4,))
    actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=0.001)
    agent = PPO(
        actor_network,
        actor_optimizer,
        critic_network,
        critic_optimizer,
        shuffle_seed=0,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    actor_before = {
        name: value.detach().clone()
        for name, value in actor_network.state_dict().items()
    }

    result = train_episodes(
        env,
        agent,
        environment_seeds=(0, 1),
        action_seeds=(10, 11),
        discount=0.99,
        gae_lambda=0.95,
        update_epochs=2,
        minibatch_size=4,
    )
    env.close()

    assert len(result.episodes) == 2
    assert all(episode.episode_length == 3 for episode in result.episodes)
    assert all(episode.truncated for episode in result.episodes)
    assert all(not episode.terminated for episode in result.episodes)
    # Six samples in minibatches of four: two minibatches in each of two epochs.
    assert result.minibatch_count == 4
    assert all(
        math.isfinite(value)
        for value in (
            result.update.actor_loss,
            result.update.critic_loss,
            result.update.entropy,
        )
    )
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in actor_network.state_dict().items()
    )
