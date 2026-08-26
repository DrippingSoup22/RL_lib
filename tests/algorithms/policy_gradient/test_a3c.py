from threading import Lock

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rl_lib.algorithms.policy_gradient.a3c import A3C
from rl_lib.data import EpisodeStep
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork

Observation = NDArray[np.float32]


def observation(value: float) -> Observation:
    return np.asarray([value], dtype=np.float32)


def fill_model(model: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def make_agent() -> A3C:
    actor_model = DiscretePolicyNetwork(1, 2, hidden_sizes=())
    shared_actor_model = DiscretePolicyNetwork(1, 2, hidden_sizes=())
    critic_model = StateValueNetwork(1, hidden_sizes=())
    shared_critic_model = StateValueNetwork(1, hidden_sizes=())

    fill_model(actor_model, 1.0)
    fill_model(critic_model, 1.0)
    fill_model(shared_actor_model, 0.0)
    fill_model(shared_critic_model, 0.0)

    return A3C(
        actor_model,
        shared_actor_model,
        torch.optim.SGD(shared_actor_model.parameters(), lr=0.1),
        critic_model,
        shared_critic_model,
        torch.optim.SGD(shared_critic_model.parameters(), lr=0.1),
        Lock(),
    )


def assert_models_equal(first: torch.nn.Module, second: torch.nn.Module) -> None:
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(),
            second.parameters(),
            strict=True,
        )
    )


def test_a3c_initially_synchronizes_local_models_from_shared_models() -> None:
    agent = make_agent()

    assert_models_equal(agent.actor_model, agent.shared_actor_model)
    assert_models_equal(agent.critic_model, agent.shared_critic_model)
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for model in (agent.actor_model, agent.critic_model)
        for parameter in model.parameters()
    )


def test_a3c_applies_local_gradients_to_shared_models_and_resynchronizes() -> None:
    agent = make_agent()
    state = observation(1.0)
    with torch.no_grad():
        probability_before = torch.softmax(
            agent.shared_actor_model(torch.as_tensor(state)), dim=-1
        )[0].item()

    actor_loss, critic_loss = agent.update(
        (EpisodeStep(state, 0, 1.0),),
        observation(0.0),
        terminated=True,
    )

    with torch.no_grad():
        probability_after = torch.softmax(
            agent.shared_actor_model(torch.as_tensor(state)), dim=-1
        )[0].item()
        shared_value = agent.shared_critic_model(torch.as_tensor(state)).item()

    assert actor_loss == pytest.approx(np.log(2.0))
    assert critic_loss == pytest.approx(0.5)
    assert probability_after > probability_before
    assert shared_value == pytest.approx(0.2)
    assert_models_equal(agent.actor_model, agent.shared_actor_model)
    assert_models_equal(agent.critic_model, agent.shared_critic_model)


def test_a3c_requires_separate_local_and_shared_parameters() -> None:
    actor_model = DiscretePolicyNetwork(1, 2, hidden_sizes=())
    critic_model = StateValueNetwork(1, hidden_sizes=())
    shared_critic_model = StateValueNetwork(1, hidden_sizes=())

    with pytest.raises(ValueError, match="separate parameters"):
        A3C(
            actor_model,
            actor_model,
            torch.optim.SGD(actor_model.parameters(), lr=0.1),
            critic_model,
            shared_critic_model,
            torch.optim.SGD(shared_critic_model.parameters(), lr=0.1),
            Lock(),
        )
