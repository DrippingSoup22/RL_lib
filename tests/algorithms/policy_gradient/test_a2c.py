from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rl_lib.algorithms.policy_gradient import A2C
from rl_lib.data import EpisodeStep
from rl_lib.models import (
    DiscretePolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)

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
    actor_model = DiscretePolicyNetwork(1, 2, hidden_sizes=())
    actor_layer = actor_model.network[0]
    assert isinstance(actor_layer, torch.nn.Linear)
    critic_model = StateValueNetwork(1, hidden_sizes=())
    critic_layer = critic_model.network[0]
    assert isinstance(critic_layer, torch.nn.Linear)
    with torch.no_grad():
        actor_layer.weight.zero_()
        actor_layer.bias.zero_()
        critic_layer.weight.zero_()
        critic_layer.bias.fill_(initial_value)

    return A2C(
        actor_model,
        torch.optim.SGD(actor_model.parameters(), lr=actor_learning_rate),
        critic_model,
        torch.optim.SGD(critic_model.parameters(), lr=critic_learning_rate),
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
            agent.actor_model(torch.as_tensor(state)), dim=-1
        )[0].item()

    actor_loss, critic_loss = agent.update(
        one_step(),
        observation(0.0),
        terminated=True,
    )

    with torch.no_grad():
        probability_after = torch.softmax(
            agent.actor_model(torch.as_tensor(state)), dim=-1
        )[0].item()
        value_after = agent.critic_model(torch.as_tensor(state)).item()
    assert actor_loss == pytest.approx(np.log(2.0))
    assert critic_loss == pytest.approx(0.5)
    assert probability_after > probability_before
    assert value_after == pytest.approx(0.2)


def test_continuous_a2c_updates_from_stored_latent_action() -> None:
    actor_model = GaussianPolicyNetwork(1, 1, hidden_sizes=(), initial_std=0.5)
    critic_model = StateValueNetwork(1, hidden_sizes=())
    with torch.no_grad():
        actor_model.mean_network[0].weight.zero_()
        actor_model.mean_network[0].bias.zero_()
        critic_model.network[0].weight.zero_()
        critic_model.network[0].bias.zero_()
    agent = A2C(
        actor_model,
        torch.optim.SGD(actor_model.parameters(), lr=0.1),
        critic_model,
        torch.optim.SGD(critic_model.parameters(), lr=0.1),
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
    assert actor_model.mean_network[0](torch.tensor([1.0])).item() > 0.0
    assert agent.critic_model(torch.tensor([1.0])).item() == pytest.approx(0.2)


def test_a2c_uses_n_step_return_targets() -> None:
    agent = make_agent(
        actor_learning_rate=0.0,
        critic_learning_rate=0.0,
        discount=0.5,
    )
    steps = (
        EpisodeStep(observation(0.0), 0, 1.0),
        EpisodeStep(observation(0.0), 1, 2.0),
    )

    _, critic_loss = agent.update(
        steps,
        observation(0.0),
        terminated=True,
    )

    # Both targets are 2: the last reward is 2 and the first is 1 + 0.5 * 2.
    assert critic_loss == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("terminated", "expected_critic_loss"),
    ((True, 0.5), (False, 0.0)),
)
def test_a2c_bootstraps_only_without_true_termination(
    terminated: bool,
    expected_critic_loss: float,
) -> None:
    agent = make_agent(
        initial_value=2.0,
        actor_learning_rate=0.0,
        critic_learning_rate=0.0,
        discount=0.5,
    )

    _, critic_loss = agent.update(
        one_step(),
        observation(0.0),
        terminated=terminated,
    )

    # Termination targets reward=1. Otherwise the target is 1 + 0.5 * 2 = 2.
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


def test_a2c_requires_matching_observation_sizes() -> None:
    actor_model = DiscretePolicyNetwork(1, 2, hidden_sizes=())
    critic_model = StateValueNetwork(2, hidden_sizes=())

    with pytest.raises(ValueError, match="same observation size"):
        A2C(
            actor_model,
            torch.optim.SGD(actor_model.parameters(), lr=0.1),
            critic_model,
            torch.optim.SGD(critic_model.parameters(), lr=0.1),
        )


@pytest.mark.parametrize("discount", (-0.1, 1.1, np.nan, np.inf, -np.inf))
def test_a2c_rejects_invalid_discount(discount: float) -> None:
    with pytest.raises(ValueError, match="Discount"):
        make_agent(discount=discount)


@pytest.mark.parametrize(
    "entropy_coefficient",
    (-0.1, np.nan, np.inf, -np.inf),
)
def test_a2c_rejects_invalid_entropy_coefficient(
    entropy_coefficient: float,
) -> None:
    with pytest.raises(ValueError, match="Entropy coefficient"):
        make_agent(entropy_coefficient=entropy_coefficient)


def test_a2c_rejects_an_empty_rollout() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_agent().update((), observation(0.0), terminated=True)


@pytest.mark.parametrize(
    "final_state",
    (
        np.zeros(2, dtype=np.float32),
        observation(np.nan),
        observation(np.inf),
        observation(-np.inf),
    ),
)
def test_a2c_rejects_an_invalid_final_state(final_state: Observation) -> None:
    with pytest.raises(ValueError, match="final state"):
        make_agent().update(one_step(), final_state, terminated=False)


@pytest.mark.parametrize("action", (-1, 2))
def test_a2c_rejects_an_action_outside_the_action_space(action: int) -> None:
    with pytest.raises(ValueError, match="action space"):
        make_agent().update(
            one_step(action=action),
            observation(0.0),
            terminated=True,
        )


@pytest.mark.parametrize("action", (True, 0.5))
def test_a2c_rejects_a_noninteger_action(action: object) -> None:
    malformed_steps = (EpisodeStep(observation(1.0), cast(int, action), 1.0),)

    with pytest.raises(ValueError, match="integers"):
        make_agent().update(
            malformed_steps,
            observation(0.0),
            terminated=True,
        )
