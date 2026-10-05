import dataclasses
import math

import pytest
import torch

from rl_lib.algorithms.policy_gradient import PPO, PPOUpdateSummary
from rl_lib.networks import (
    CategoricalPolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)
from rl_lib.policies import CategoricalPolicy

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
LOG_HALF = -math.log(2.0)  # log-probability of each action under the zero actor


def linear_critic() -> StateValueNetwork:
    """Value 0.5 + x0 + 2 x1."""
    critic_network = StateValueNetwork(2, hidden_sizes=())
    with torch.no_grad():
        critic_network.network[0].weight.copy_(torch.tensor([[1.0, 2.0]]))
        critic_network.network[0].bias.fill_(0.5)
    return critic_network


def make_agent(*, device="cpu", **settings) -> PPO:
    """Categorical PPO whose actor starts with equal probabilities."""
    actor_network = CategoricalPolicyNetwork(2, 2, hidden_sizes=())
    with torch.no_grad():
        actor_network.network[0].weight.zero_()
        actor_network.network[0].bias.zero_()
    actor_network, critic_network = actor_network.to(device), linear_critic().to(device)
    return PPO(
        actor_network,
        torch.optim.SGD(actor_network.parameters(), lr=0.1),
        critic_network,
        torch.optim.SGD(critic_network.parameters(), lr=0.1),
        shuffle_seed=0,
        **settings,
    )


def make_continuous_agent(*, device="cpu") -> PPO:
    """Continuous PPO whose Gaussian mean equals the observation."""
    actor_network = GaussianPolicyNetwork(2, 2, hidden_sizes=(), initial_std=0.5)
    with torch.no_grad():
        actor_network.mean_network[0].weight.copy_(torch.eye(2))
        actor_network.mean_network[0].bias.zero_()
    actor_network, critic_network = actor_network.to(device), linear_critic().to(device)
    return PPO(
        actor_network,
        torch.optim.SGD(actor_network.parameters(), lr=0.1),
        critic_network,
        torch.optim.SGD(critic_network.parameters(), lr=0.1),
        shuffle_seed=0,
        action_low=[-2.0, 0.0],
        action_high=[2.0, 10.0],
    )


def minibatch(agent, policy_actions, old_log_probabilities, advantages, targets):
    """One minibatch step on observations of zeros; returns the measurements."""
    measurements = agent._update_minibatch(
        torch.zeros(len(advantages), 2),
        torch.as_tensor(policy_actions),
        torch.as_tensor(old_log_probabilities),
        torch.tensor(advantages),
        torch.tensor(targets),
    )
    names = [field.name for field in dataclasses.fields(PPOUpdateSummary)]
    return dict(zip(names, measurements.tolist(), strict=True))


@pytest.mark.parametrize("make", (make_agent, make_continuous_agent))
def test_sampling_records_what_learning_needs(make) -> None:
    agent = make()
    observations = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.5, 2.0]])

    torch.manual_seed(0)
    sample = agent.sample_action(observations)

    log_probabilities, _ = agent.policy.evaluate_actions(
        observations, sample.policy_action
    )
    torch.testing.assert_close(sample.log_probability, log_probabilities.detach())
    torch.testing.assert_close(sample.value, torch.tensor([0.5, -0.5, 5.0]))
    assert not sample.log_probability.requires_grad
    if isinstance(agent.policy, CategoricalPolicy):
        assert torch.equal(sample.environment_action, sample.policy_action)
    else:
        # The environment gets the bounded action; learning keeps the latent one.
        torch.testing.assert_close(
            sample.environment_action,
            agent.policy.scale * torch.tanh(sample.policy_action) + agent.policy.bias,
        )
    torch.testing.assert_close(
        agent.select_action(observations, deterministic=True),
        agent.policy.deterministic_action(observations),
    )
    torch.testing.assert_close(agent.state_value(observations), sample.value)


def test_minibatch_clips_only_advantageous_policy_changes() -> None:
    # (advantage, new / old probability ratio, expected actor loss), clip 0.2.
    for advantage, ratio, expected_actor_loss in (
        (1.0, 2.0, -1.2),
        (-1.0, 0.5, 0.8),
        (1.0, 0.5, -0.5),
        (-1.0, 2.0, 2.0),
    ):
        measured = minibatch(
            make_agent(clip_ratio=0.2),
            [0],
            [LOG_HALF - math.log(ratio)],
            [advantage],
            [0.5],
        )

        assert measured["actor_loss"] == pytest.approx(expected_actor_loss)
        assert measured["critic_loss"] == pytest.approx(0.0)
        assert measured["approximate_kl"] == pytest.approx(
            ratio - 1.0 - math.log(ratio)
        )
        assert measured["clip_fraction"] == float(abs(ratio - 1.0) > 0.2)


def test_minibatch_moves_actor_and_critic_towards_fixed_targets() -> None:
    agent = make_agent()
    observation = torch.zeros(1, 2)
    probability_before = torch.softmax(agent.actor_network(observation), -1)[0, 0]

    measured = minibatch(agent, [0], [LOG_HALF], [1.0], [1.5])

    probability_after = torch.softmax(agent.actor_network(observation), -1)[0, 0]
    assert measured["actor_loss"] == pytest.approx(-1.0)
    assert measured["critic_loss"] == pytest.approx(0.5)
    assert probability_after > probability_before
    assert agent.state_value(observation).item() == pytest.approx(0.6)

    # Continuous PPO re-evaluates the stored latent sample, so an unchanged
    # policy has a probability ratio of exactly 1.
    agent = make_continuous_agent()
    torch.manual_seed(17)
    sample = agent.sample_action(observation)
    bias_before = agent.actor_network.mean_network[0].bias.detach().clone()

    measured = minibatch(
        agent, sample.policy_action, sample.log_probability, [1.0], [1.5]
    )

    assert measured["actor_loss"] == pytest.approx(-1.0)
    assert measured["critic_loss"] == pytest.approx(0.5)
    assert not torch.equal(agent.actor_network.mean_network[0].bias, bias_before)


def test_minibatch_subtracts_the_entropy_bonus() -> None:
    measured = minibatch(
        make_agent(entropy_coefficient=0.1), [0], [LOG_HALF], [0.0], [0.5]
    )

    assert measured["entropy"] == pytest.approx(math.log(2.0))
    assert measured["actor_loss"] == pytest.approx(-0.1 * math.log(2.0))


def test_minibatch_clips_both_networks_gradients_when_configured(monkeypatch) -> None:
    limits = []
    monkeypatch.setattr(
        "rl_lib.algorithms.policy_gradient.ppo.clip_gradients",
        lambda _parameters, limit: limits.append(limit),
    )

    minibatch(make_agent(max_gradient_norm=0.5), [0], [LOG_HALF], [1.0], [1.5])

    assert limits == [0.5, 0.5]


def test_update_normalizes_advantages_once_and_uses_every_sample_each_epoch(
    monkeypatch,
) -> None:
    agent = make_agent()
    minibatches = []

    def record_minibatch(observations, _actions, _old, advantages, _targets):
        minibatches.append((observations[:, 0].tolist(), advantages.tolist()))
        return torch.arange(5.0)

    monkeypatch.setattr(agent, "_update_minibatch", record_minibatch)

    summary = agent.update(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.zeros(3),
        update_epochs=2,
        minibatch_size=2,
    )

    # Each epoch: a minibatch of two, then one of one. The summary averages
    # the four minibatches' measurements, in its field order.
    assert len(minibatches) == 4
    assert summary == PPOUpdateSummary(0.0, 1.0, 2.0, 3.0, 4.0)
    normalized_advantage = {0: -math.sqrt(1.5), 1: 0.0, 2: math.sqrt(1.5)}
    for epoch in (minibatches[:2], minibatches[2:]):
        samples = [
            (int(index), advantage)
            for indices, advantages in epoch
            for index, advantage in zip(indices, advantages, strict=True)
        ]
        assert sorted(index for index, _ in samples) == [0, 1, 2]
        for index, advantage in samples:
            assert advantage == pytest.approx(normalized_advantage[index])

    # Nearly constant advantages are kept rather than divided by almost zero.
    minibatches.clear()
    agent.update(
        torch.zeros(2, 2),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2),
        torch.tensor([2.0, 2.0]),
        torch.zeros(2),
        update_epochs=1,
        minibatch_size=2,
    )
    assert minibatches[0][1] == [2.0, 2.0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_full_update_learns_from_sampled_data_on_the_networks_device(
    device,
) -> None:
    agent = make_continuous_agent(device=device)
    observations = torch.randn(8, 2, device=device)
    sample = agent.sample_action(observations)
    parameters_before = [
        parameter.detach().clone() for parameter in agent.actor_network.parameters()
    ]

    summary = agent.update(
        observations,
        sample.policy_action,
        sample.log_probability,
        torch.randn(8, device=device),
        sample.value + 1.0,
        update_epochs=2,
        minibatch_size=4,
    )

    assert all(math.isfinite(value) for value in dataclasses.astuple(summary))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before, agent.actor_network.parameters(), strict=True
        )
    )


def test_constructor_rejects_invalid_settings() -> None:
    categorical = CategoricalPolicyNetwork(2, 2)
    gaussian = GaussianPolicyNetwork(2, 2)
    critic = StateValueNetwork(2)
    bounds = {"action_low": [-1.0, -1.0], "action_high": [1.0, 1.0]}
    for actor_network, critic_network, settings in (
        (categorical, StateValueNetwork(3), {}),
        (categorical, critic, {"clip_ratio": 1.0}),
        (categorical, critic, {"entropy_coefficient": -0.1}),
        (categorical, critic, {"max_gradient_norm": 0.0}),
        (categorical, critic, bounds),
        (gaussian, critic, {}),
    ):
        with pytest.raises(ValueError):
            PPO(
                actor_network,
                torch.optim.SGD(actor_network.parameters(), lr=0.1),
                critic_network,
                torch.optim.SGD(critic_network.parameters(), lr=0.1),
                **settings,
            )
