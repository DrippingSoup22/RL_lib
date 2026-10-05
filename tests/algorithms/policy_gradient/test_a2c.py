import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rl_lib.algorithms.policy_gradient import A2C
from rl_lib.networks import (
    CategoricalPolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)
from rl_lib.trajectories import EpisodeStep

Observation = NDArray[np.float32]


def observation(value: float) -> Observation:
    return np.asarray([value], dtype=np.float32)


def make_agent(
    *,
    initial_value: float = 0.0,
    actor_learning_rate: float = 0.1,
    critic_learning_rate: float = 0.1,
    discount: float = 1.0,
    entropy_coefficient: float = 0.0,
) -> A2C:
    actor_network = CategoricalPolicyNetwork(1, 2, hidden_sizes=())
    actor_layer = actor_network.network[0]
    assert isinstance(actor_layer, torch.nn.Linear)
    critic_network = StateValueNetwork(1, hidden_sizes=())
    critic_layer = critic_network.network[0]
    assert isinstance(critic_layer, torch.nn.Linear)
    with torch.no_grad():
        actor_layer.weight.zero_()
        actor_layer.bias.zero_()
        critic_layer.weight.zero_()
        critic_layer.bias.fill_(initial_value)

    return A2C(
        actor_network,
        torch.optim.SGD(actor_network.parameters(), lr=actor_learning_rate),
        critic_network,
        torch.optim.SGD(critic_network.parameters(), lr=critic_learning_rate),
        discount=discount,
        entropy_coefficient=entropy_coefficient,
    )


def one_step(*, action: int = 0, reward: float = 1.0) -> tuple[EpisodeStep]:
    return (EpisodeStep(observation(1.0), action, reward),)


def test_a2c_updates_actor_and_critic_from_a_positive_advantage() -> None:
    agent = make_agent()
    state = observation(1.0)
    with torch.no_grad():
        probability_before = torch.softmax(
            agent.actor_network(torch.as_tensor(state)), dim=-1
        )[0].item()

    actor_loss, critic_loss = agent.update(
        one_step(),
        observation(0.0),
        terminated=True,
    )

    with torch.no_grad():
        probability_after = torch.softmax(
            agent.actor_network(torch.as_tensor(state)), dim=-1
        )[0].item()
        value_after = agent.critic_network(torch.as_tensor(state)).item()
    assert actor_loss == pytest.approx(np.log(2.0))
    assert critic_loss == pytest.approx(0.5)
    assert probability_after > probability_before
    assert value_after == pytest.approx(0.2)


def test_continuous_a2c_updates_from_stored_latent_action() -> None:
    actor_network = GaussianPolicyNetwork(1, 1, hidden_sizes=(), initial_std=0.5)
    critic_network = StateValueNetwork(1, hidden_sizes=())
    with torch.no_grad():
        actor_network.mean_network[0].weight.zero_()
        actor_network.mean_network[0].bias.zero_()
        critic_network.network[0].weight.zero_()
        critic_network.network[0].bias.zero_()
    agent = A2C(
        actor_network,
        torch.optim.SGD(actor_network.parameters(), lr=0.1),
        critic_network,
        torch.optim.SGD(critic_network.parameters(), lr=0.1),
        action_low=[-1.0],
        action_high=[1.0],
    )
    latent_action = np.asarray([0.5], dtype=np.float32)
    steps = (
        EpisodeStep(
            observation(1.0),
            np.tanh(latent_action).astype(np.float32),
            1.0,
            policy_action=latent_action,
        ),
    )

    actor_loss, critic_loss = agent.update(
        steps,
        observation(0.0),
        terminated=True,
    )

    assert np.isfinite(actor_loss)
    assert critic_loss == pytest.approx(0.5)
    assert actor_network.mean_network[0](torch.tensor([1.0])).item() > 0.0
    assert agent.critic_network(torch.tensor([1.0])).item() == pytest.approx(0.2)


def test_a2c_critic_targets_are_n_step_returns_bootstrapped_only_if_not_ended():
    steps = (
        EpisodeStep(observation(0.0), 0, 1.0),
        EpisodeStep(observation(0.0), 1, 2.0),
    )
    frozen = {"actor_learning_rate": 0.0, "critic_learning_rate": 0.0}

    # Both targets are 2: the last reward is 2 and the first is 1 + 0.5 * 2.
    agent = make_agent(discount=0.5, **frozen)
    _, critic_loss = agent.update(steps, observation(0.0), terminated=True)
    assert critic_loss == pytest.approx(2.0)

    # With values of 2, termination targets the reward 1; otherwise the
    # target bootstraps to 1 + 0.5 * 2 = 2.
    for terminated, expected_critic_loss in ((True, 0.5), (False, 0.0)):
        agent = make_agent(initial_value=2.0, discount=0.5, **frozen)
        _, critic_loss = agent.update(
            one_step(), observation(0.0), terminated=terminated
        )
        assert critic_loss == pytest.approx(expected_critic_loss)


def test_a2c_optionally_subtracts_an_entropy_bonus() -> None:
    entropy_coefficient = 0.1
    agent = make_agent(
        actor_learning_rate=0.0,
        critic_learning_rate=0.0,
        entropy_coefficient=entropy_coefficient,
    )

    actor_loss, critic_loss = agent.update(
        one_step(reward=0.0),
        observation(0.0),
        terminated=True,
    )

    assert actor_loss == pytest.approx(-entropy_coefficient * np.log(2.0))
    assert critic_loss == pytest.approx(0.0)


def test_a2c_constructor_rejects_invalid_settings() -> None:
    actor_network = CategoricalPolicyNetwork(1, 2, hidden_sizes=())
    critic_network = StateValueNetwork(2, hidden_sizes=())
    with pytest.raises(ValueError, match="same observation size"):
        A2C(
            actor_network,
            torch.optim.SGD(actor_network.parameters(), lr=0.1),
            critic_network,
            torch.optim.SGD(critic_network.parameters(), lr=0.1),
        )
    for settings, message in (
        ({"discount": 1.1}, "Discount"),
        ({"discount": np.nan}, "Discount"),
        ({"entropy_coefficient": -0.1}, "Entropy coefficient"),
    ):
        with pytest.raises(ValueError, match=message):
            make_agent(**settings)
