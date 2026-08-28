import math

import pytest
import torch

from experiments.policy_gradient.run import (
    make_agent,
    make_environment,
    make_learning_rate_scheduler,
    parse_config,
)
from experiments.policy_gradient.runners.a2c import train_episode as train_a2c_episode
from rl_lib.algorithms.policy_gradient import A2C, PPO
from rl_lib.models import GaussianPolicyNetwork
from rl_lib.policies import SquashedGaussianPolicy


def test_learning_rate_warms_up_then_cosine_decays_to_minimum() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.01)
    scheduler = make_learning_rate_scheduler(
        optimizer,
        training_episodes=6,
        minimum_learning_rate=0.001,
        warmup_episodes=2,
        warmup_start_factor=0.1,
    )
    learning_rates = [float(optimizer.param_groups[0]["lr"])]

    for _ in range(6):
        optimizer.zero_grad()
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))

    assert learning_rates == pytest.approx(
        (0.001, 0.0055, 0.01, 0.00868198, 0.0055, 0.00231802, 0.001)
    )


def test_policy_gradient_defaults_include_refined_optimization_and_a2c() -> None:
    config = parse_config(
        ("--preset", "quick", "--algorithm", "a2c", "--recordings", "none")
    )

    assert config.actor_learning_rate == 0.003
    assert config.critic_learning_rate == 0.01
    assert config.optimizer == "adamw"
    assert config.weight_decay == 0.0001
    assert config.warmup_episodes == 1
    assert config.rollout_steps == 5
    assert config.a3c_workers >= 1
    assert config.entropy_coefficient == 0.0
    assert config.ppo_batch_episodes == 4
    assert config.ppo_update_epochs == 4
    assert config.ppo_minibatch_size == 64
    assert config.ppo_clip_ratio == 0.2
    assert config.gae_lambda == 0.95
    assert config.algorithm == "a2c"
    assert config.seed_values == (0,)
    assert not config.diagnostics


def test_policy_gradient_a2c_options_are_configurable() -> None:
    config = parse_config(
        (
            "--preset",
            "quick",
            "--recordings",
            "none",
            "--rollout",
            "3",
            "--entropy",
            "0.02",
            "--diag",
            "--algorithm",
            "a2c",
        )
    )

    assert config.rollout_steps == 3
    assert config.entropy_coefficient == 0.02
    assert config.algorithm == "a2c"
    assert config.diagnostics


def test_policy_gradient_a3c_workers_are_configurable() -> None:
    config = parse_config(
        (
            "--preset",
            "quick",
            "--recordings",
            "none",
            "--algorithm",
            "a3c",
            "--workers",
            "2",
        )
    )

    assert config.algorithm == "a3c"
    assert config.a3c_workers == 2


def test_policy_gradient_ppo_options_are_configurable() -> None:
    config = parse_config(
        (
            "--preset",
            "quick",
            "--recordings",
            "none",
            "--algorithm",
            "ppo",
            "--batch-episodes",
            "2",
            "--update-epochs",
            "3",
            "--minibatch-size",
            "8",
            "--clip-ratio",
            "0.1",
            "--gae-lambda",
            "0.9",
        )
    )

    assert config.algorithm == "ppo"
    assert config.ppo_batch_episodes == 2
    assert config.ppo_update_epochs == 3
    assert config.ppo_minibatch_size == 8
    assert config.ppo_clip_ratio == 0.1
    assert config.gae_lambda == 0.9


def test_make_agent_builds_configured_ppo() -> None:
    env = make_environment("CartPole-v1", max_episode_steps=1)
    agent = make_agent(
        "ppo",
        env,
        actor_learning_rate=0.001,
        critic_learning_rate=0.002,
        optimizer_name="adam",
        weight_decay=0.0,
        discount=0.99,
        entropy_coefficient=0.01,
        hidden_sizes=(4,),
        seed=0,
        ppo_clip_ratio=0.1,
    )

    assert isinstance(agent, PPO)
    assert agent.clip_ratio == 0.1
    assert agent.entropy_coefficient == 0.01
    env.close()


def test_make_agent_builds_continuous_ppo_from_box_action_space() -> None:
    env = make_environment("MountainCarContinuous-v0", max_episode_steps=1)
    agent = make_agent(
        "ppo",
        env,
        actor_learning_rate=0.001,
        critic_learning_rate=0.002,
        optimizer_name="adam",
        weight_decay=0.0,
        discount=0.99,
        entropy_coefficient=0.01,
        hidden_sizes=(4,),
        seed=0,
        ppo_clip_ratio=0.1,
    )

    assert isinstance(agent.actor_model, GaussianPolicyNetwork)
    assert isinstance(agent.policy, SquashedGaussianPolicy)
    assert agent.actor_model.action_size == 1
    env.close()


@pytest.mark.parametrize(
    "algorithm",
    ("reinforce", "reinforce_with_baseline", "a2c", "a3c", "ppo"),
)
def test_make_agent_builds_continuous_policy_for_every_algorithm(
    algorithm: str,
) -> None:
    env = make_environment("MountainCarContinuous-v0", max_episode_steps=1)
    agent = make_agent(
        algorithm,
        env,
        actor_learning_rate=0.001,
        critic_learning_rate=0.002,
        optimizer_name="adam",
        weight_decay=0.0,
        discount=0.99,
        entropy_coefficient=0.01,
        hidden_sizes=(4,),
        seed=0,
    )

    assert isinstance(agent.actor_model, GaussianPolicyNetwork)
    assert isinstance(agent.policy, SquashedGaussianPolicy)
    env.close()


def test_policy_gradient_seed_base_resolves_visible_trial_values() -> None:
    config = parse_config(
        (
            "--preset",
            "tuning",
            "--seeds",
            "3",
            "--seed-base",
            "120",
            "--recordings",
            "none",
            "--algorithm",
            "a2c",
        )
    )

    assert config.seed_values == (120, 121, 122)


def test_a2c_training_updates_at_rollout_boundaries_and_truncation() -> None:
    env = make_environment("CartPole-v1", max_episode_steps=3)
    agent = make_agent(
        "a2c",
        env,
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        optimizer_name="adam",
        weight_decay=0.0,
        discount=0.99,
        entropy_coefficient=0.0,
        hidden_sizes=(4,),
        seed=0,
    )
    assert isinstance(agent, A2C)

    result = train_a2c_episode(
        env,
        agent,
        rollout_steps=2,
        environment_seed=0,
        action_seed=0,
        collect_diagnostics=True,
    )

    assert result.episode_length == 3
    assert result.truncated
    assert not result.terminated
    assert result.updates == 2
    assert result.critic_loss is not None
    assert math.isfinite(result.actor_loss)
    assert math.isfinite(result.critic_loss)
    assert result.policy_entropy is not None
    assert math.isfinite(result.policy_entropy)
    env.close()


def test_continuous_a2c_training_updates_at_rollout_boundaries() -> None:
    env = make_environment("MountainCarContinuous-v0", max_episode_steps=3)
    agent = make_agent(
        "a2c",
        env,
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        optimizer_name="adam",
        weight_decay=0.0,
        discount=0.99,
        entropy_coefficient=0.0,
        hidden_sizes=(4,),
        seed=0,
    )
    assert isinstance(agent, A2C)

    result = train_a2c_episode(
        env,
        agent,
        rollout_steps=2,
        environment_seed=0,
        action_seed=0,
        collect_diagnostics=True,
    )

    assert result.episode_length == 3
    assert result.truncated
    assert result.updates == 2
    assert math.isfinite(result.actor_loss)
    assert result.critic_loss is not None and math.isfinite(result.critic_loss)
    assert result.policy_entropy is not None
    assert math.isfinite(result.policy_entropy)
    env.close()
