from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rl_lib.algorithms.policy_gradient import Reinforce, ReinforceWithBaseline
from rl_lib.data import Episode, EpisodeStep
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork

Observation = NDArray[np.float32]


def observation(value: float) -> Observation:
    return np.asarray([value], dtype=np.float32)


def make_agent(*, discount: float = 1.0) -> Reinforce:
    model = DiscretePolicyNetwork(1, 2, hidden_sizes=())
    layer = model.network[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return Reinforce(model, optimizer, discount)


def make_baseline_agent(initial_value: float) -> ReinforceWithBaseline:
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

    actor_optimizer = torch.optim.SGD(actor_model.parameters(), lr=0.1)
    critic_optimizer = torch.optim.SGD(critic_model.parameters(), lr=0.1)
    return ReinforceWithBaseline(
        actor_model,
        actor_optimizer,
        critic_model,
        critic_optimizer,
    )


def one_step_episode(
    *,
    state: Observation | None = None,
    action: int = 0,
    reward: float = 1.0,
) -> Episode[Observation]:
    state = observation(1.0) if state is None else state
    return Episode(
        steps=(EpisodeStep(state, action, reward),),
        final_state=observation(0.0),
        terminated=True,
        truncated=False,
    )


def test_reinforce_increases_probability_of_a_rewarded_action() -> None:
    agent = make_agent()
    state = observation(1.0)
    with torch.no_grad():
        probability_before = torch.softmax(
            agent.actor_model(torch.as_tensor(state)), dim=-1
        )[0].item()

    agent.update(one_step_episode(state=state, action=0, reward=1.0))

    with torch.no_grad():
        probability_after = torch.softmax(
            agent.actor_model(torch.as_tensor(state)), dim=-1
        )[0].item()
    assert probability_after > probability_before


def test_reinforce_uses_reward_to_go_and_outer_discount() -> None:
    agent = make_agent(discount=0.5)
    state = observation(1.0)
    episode = Episode(
        steps=(
            EpisodeStep(state, 0, 1.0),
            EpisodeStep(state, 0, 2.0),
        ),
        final_state=observation(0.0),
        terminated=True,
        truncated=False,
    )

    loss = agent.update(episode)

    layer = agent.actor_model.network[0]
    assert isinstance(layer, torch.nn.Linear)
    assert loss == pytest.approx(3.0 * np.log(2.0))
    torch.testing.assert_close(layer.weight, torch.tensor([[0.15], [-0.15]]))
    torch.testing.assert_close(layer.bias, torch.tensor([0.15, -0.15]))


@pytest.mark.parametrize(
    (
        "initial_value",
        "expected_loss",
        "expected_action_probability",
        "expected_value",
    ),
    (
        (0.0, np.log(2.0), 0.549833997, 0.2),
        (2.0, -np.log(2.0), 0.450166003, 1.8),
    ),
)
def test_reinforce_with_baseline_updates_actor_and_critic(
    initial_value: float,
    expected_loss: float,
    expected_action_probability: float,
    expected_value: float,
) -> None:
    agent = make_baseline_agent(initial_value)
    state = observation(1.0)

    loss = agent.update(one_step_episode(state=state, action=0, reward=1.0))

    with torch.no_grad():
        action_probability = torch.softmax(
            agent.actor_model(torch.as_tensor(state)), dim=-1
        )[0].item()
        value = agent.critic_model(torch.as_tensor(state)).item()
    assert loss == pytest.approx(expected_loss)
    assert action_probability == pytest.approx(expected_action_probability)
    assert value == pytest.approx(expected_value)


def test_reinforce_with_baseline_requires_matching_observation_sizes() -> None:
    actor_model = DiscretePolicyNetwork(1, 2, hidden_sizes=())
    critic_model = StateValueNetwork(2, hidden_sizes=())

    with pytest.raises(ValueError, match="same observation size"):
        ReinforceWithBaseline(
            actor_model,
            torch.optim.SGD(actor_model.parameters(), lr=0.1),
            critic_model,
            torch.optim.SGD(critic_model.parameters(), lr=0.1),
        )


@pytest.mark.parametrize("discount", (-0.1, 1.1, np.nan, np.inf, -np.inf))
def test_reinforce_rejects_invalid_discount(discount: float) -> None:
    with pytest.raises(ValueError, match="Discount"):
        make_agent(discount=discount)


def test_reinforce_rejects_an_empty_episode() -> None:
    episode: Episode[Observation] = Episode(
        steps=(),
        final_state=observation(0.0),
        terminated=True,
        truncated=False,
    )

    with pytest.raises(ValueError, match="at least one step"):
        make_agent().update(episode)


@pytest.mark.parametrize("action", (-1, 2))
def test_reinforce_rejects_an_action_outside_the_action_space(action: int) -> None:
    with pytest.raises(ValueError, match="action space"):
        make_agent().update(one_step_episode(action=action))


@pytest.mark.parametrize("action", (True, 0.5))
def test_reinforce_rejects_a_noninteger_action(action: object) -> None:
    malformed_episode = Episode(
        steps=(EpisodeStep(observation(1.0), cast(int, action), 1.0),),
        final_state=observation(0.0),
        terminated=True,
        truncated=False,
    )

    with pytest.raises(ValueError, match="integers"):
        make_agent().update(malformed_episode)


@pytest.mark.parametrize("reward", (np.nan, np.inf, -np.inf))
def test_reinforce_rejects_a_nonfinite_reward(reward: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_agent().update(one_step_episode(reward=reward))


def test_reinforce_rejects_an_observation_with_the_wrong_size() -> None:
    with pytest.raises(ValueError, match="model input size"):
        make_agent().update(one_step_episode(state=np.zeros(2, dtype=np.float32)))


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_reinforce_rejects_a_nonfinite_observation(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_agent().update(one_step_episode(state=observation(value)))
