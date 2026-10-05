import math

import pytest
import torch

from experiments.policy_gradient.runners.a3c import (
    scheduled_learning_rate,
    train_episodes,
)
from rl_lib.networks import (
    CategoricalPolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)


def test_a3c_learning_rate_matches_linear_warmup_and_cosine_decay() -> None:
    learning_rates = [
        scheduled_learning_rate(
            episode,
            total_episodes=6,
            peak_learning_rate=0.01,
            minimum_learning_rate=0.001,
            warmup_episodes=2,
            warmup_start_factor=0.1,
        )
        for episode in range(6)
    ]

    assert learning_rates == pytest.approx(
        (0.001, 0.0055, 0.01, 0.00868198, 0.0055, 0.00231802)
    )


def test_a3c_two_workers_share_updates_on_cartpole() -> None:
    torch.manual_seed(0)
    actor_network = CategoricalPolicyNetwork(4, 2, hidden_sizes=(4,))
    critic_network = StateValueNetwork(4, hidden_sizes=(4,))
    actor_optimizer = torch.optim.AdamW(actor_network.parameters(), lr=0.001)
    critic_optimizer = torch.optim.AdamW(critic_network.parameters(), lr=0.001)
    actor_before = {
        name: value.detach().clone()
        for name, value in actor_network.state_dict().items()
    }

    results = train_episodes(
        "CartPole-v1",
        actor_network,
        actor_optimizer,
        critic_network,
        critic_optimizer,
        episode_start=0,
        episode_count=4,
        total_episodes=4,
        workers=2,
        rollout_steps=2,
        actor_learning_rate=0.001,
        actor_minimum_learning_rate=0.0001,
        critic_learning_rate=0.001,
        critic_minimum_learning_rate=0.0001,
        warmup_episodes=0,
        warmup_start_factor=0.1,
        discount=0.99,
        entropy_coefficient=0.0,
        seed=0,
        collect_diagnostics=True,
        max_episode_steps=3,
    )

    assert [result.episode_index for result in results] == [0, 1, 2, 3]
    assert all(result.result.episode_length == 3 for result in results)
    assert all(result.result.truncated for result in results)
    assert all(result.result.updates == 2 for result in results)
    assert all(math.isfinite(result.result.actor_loss) for result in results)
    assert all(
        result.result.critic_loss is not None
        and math.isfinite(result.result.critic_loss)
        for result in results
    )
    assert all(result.result.policy_entropy is not None for result in results)

    first_actor_parameter = next(actor_network.parameters())
    first_critic_parameter = next(critic_network.parameters())
    assert actor_optimizer.state[first_actor_parameter]["step"].item() == 8
    assert critic_optimizer.state[first_critic_parameter]["step"].item() == 8
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in actor_network.state_dict().items()
    )


def test_a3c_worker_updates_continuous_gaussian_actor() -> None:
    actor_network = GaussianPolicyNetwork(2, 1, hidden_sizes=(4,))
    critic_network = StateValueNetwork(2, hidden_sizes=(4,))
    actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=0.001)
    actor_before = {
        name: value.detach().clone()
        for name, value in actor_network.state_dict().items()
    }

    results = train_episodes(
        "MountainCarContinuous-v0",
        actor_network,
        actor_optimizer,
        critic_network,
        critic_optimizer,
        episode_start=0,
        episode_count=1,
        total_episodes=1,
        workers=1,
        rollout_steps=2,
        actor_learning_rate=0.001,
        actor_minimum_learning_rate=0.0001,
        critic_learning_rate=0.001,
        critic_minimum_learning_rate=0.0001,
        warmup_episodes=0,
        warmup_start_factor=0.1,
        discount=0.99,
        entropy_coefficient=0.0,
        seed=0,
        collect_diagnostics=True,
        max_episode_steps=2,
    )

    assert len(results) == 1
    assert results[0].result.episode_length == 2
    assert results[0].result.truncated
    assert results[0].result.updates == 1
    assert math.isfinite(results[0].result.actor_loss)
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in actor_network.state_dict().items()
    )
