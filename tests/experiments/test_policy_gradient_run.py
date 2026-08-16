import math

import pytest
import torch

from experiments.policy_gradient.run import (
    ALGORITHMS,
    make_agent,
    make_environment,
    make_learning_rate_scheduler,
    parse_config,
    train_a2c_episode,
)
from rl_lib.algorithms.policy_gradient import A2C


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
    config = parse_config(("--preset", "quick", "--recordings", "none"))

    assert config.actor_learning_rate == 0.003
    assert config.critic_learning_rate == 0.01
    assert config.optimizer == "adamw"
    assert config.weight_decay == 0.0001
    assert config.warmup_episodes == 1
    assert config.a2c_rollout_steps == 5
    assert config.entropy_coefficient == 0.0
    assert config.algorithms == ALGORITHMS
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

    assert config.a2c_rollout_steps == 3
    assert config.entropy_coefficient == 0.02
    assert config.algorithms == ("a2c",)
    assert config.diagnostics


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
