from collections.abc import Callable

import numpy as np
import pytest
import torch

from rl_lib.algorithms.function_approximation import (
    SemiGradientQLearning,
    SemiGradientSARSA,
    SemiGradientTDZeroPrediction,
)
from rl_lib.models import ActionValueNetwork, StateValueNetwork

ControlAgent = SemiGradientSARSA | SemiGradientQLearning
ControlConstructor = Callable[..., ControlAgent]


def linear_predictor(
    *, discount: float = 1.0
) -> tuple[SemiGradientTDZeroPrediction, torch.nn.Linear]:
    model = StateValueNetwork(1, hidden_sizes=())
    layer = model.network[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.fill_(2.0)
        layer.bias.zero_()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return SemiGradientTDZeroPrediction(model, optimizer, discount), layer


def linear_control_agent(
    constructor: ControlConstructor,
    *,
    discount: float = 0.5,
    epsilon: float = 0.0,
) -> tuple[ControlAgent, torch.nn.Linear]:
    model = ActionValueNetwork(1, 2, hidden_sizes=())
    layer = model.network[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0], [2.0]]))
        layer.bias.zero_()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return (
        constructor(
            model,
            optimizer,
            discount=discount,
            epsilon=epsilon,
            seed=0,
        ),
        layer,
    )


def test_prediction_uses_a_fixed_bootstrap_target() -> None:
    predictor, layer = linear_predictor(discount=0.5)

    error = predictor.update([1.0], 1.0, [2.0], terminated=False)

    assert error == pytest.approx(1.0)
    torch.testing.assert_close(layer.weight, torch.tensor([[2.1]]))
    torch.testing.assert_close(layer.bias, torch.tensor([0.1]))


def test_prediction_does_not_bootstrap_after_termination() -> None:
    predictor, layer = linear_predictor(discount=0.5)

    error = predictor.update([1.0], 4.0, [100.0], terminated=True)

    assert error == pytest.approx(2.0)
    torch.testing.assert_close(layer.weight, torch.tensor([[2.2]]))
    torch.testing.assert_close(layer.bias, torch.tensor([0.2]))


def test_sarsa_uses_the_selected_next_action_as_a_fixed_target() -> None:
    agent, layer = linear_control_agent(SemiGradientSARSA)
    assert isinstance(agent, SemiGradientSARSA)

    error = agent.update([1.0], 0, 1.0, [2.0], 1, terminated=False)

    assert error == pytest.approx(2.0)
    torch.testing.assert_close(layer.weight, torch.tensor([[1.2], [2.0]]))
    torch.testing.assert_close(layer.bias, torch.tensor([0.2, 0.0]))


def test_sarsa_terminal_transition_does_not_require_a_next_action() -> None:
    agent, layer = linear_control_agent(SemiGradientSARSA)
    assert isinstance(agent, SemiGradientSARSA)

    error = agent.update([1.0], 1, 4.0, [100.0], None, terminated=True)

    assert error == pytest.approx(2.0)
    torch.testing.assert_close(layer.weight, torch.tensor([[1.0], [2.2]]))
    torch.testing.assert_close(layer.bias, torch.tensor([0.0, 0.2]))


def test_q_learning_uses_the_largest_next_action_value() -> None:
    agent, layer = linear_control_agent(SemiGradientQLearning)
    assert isinstance(agent, SemiGradientQLearning)

    error = agent.update([1.0], 0, 1.0, [2.0], terminated=False)

    assert error == pytest.approx(2.0)
    torch.testing.assert_close(layer.weight, torch.tensor([[1.2], [2.0]]))
    torch.testing.assert_close(layer.bias, torch.tensor([0.2, 0.0]))


def test_q_learning_does_not_bootstrap_after_termination() -> None:
    agent, layer = linear_control_agent(SemiGradientQLearning)
    assert isinstance(agent, SemiGradientQLearning)

    error = agent.update([1.0], 1, 4.0, [100.0], terminated=True)

    assert error == pytest.approx(2.0)
    torch.testing.assert_close(layer.weight, torch.tensor([[1.0], [2.2]]))
    torch.testing.assert_close(layer.bias, torch.tensor([0.0, 0.2]))


@pytest.mark.parametrize("constructor", (SemiGradientSARSA, SemiGradientQLearning))
def test_control_selects_an_epsilon_greedy_action(
    constructor: ControlConstructor,
) -> None:
    agent, _ = linear_control_agent(constructor, epsilon=0.0)

    assert agent.select_action([1.0]) == 1


@pytest.mark.parametrize("discount", (-0.1, 1.1, np.nan, np.inf))
def test_prediction_rejects_invalid_discount(discount: float) -> None:
    model = StateValueNetwork(1, hidden_sizes=())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError):
        SemiGradientTDZeroPrediction(model, optimizer, discount)


@pytest.mark.parametrize("constructor", (SemiGradientSARSA, SemiGradientQLearning))
@pytest.mark.parametrize(
    ("discount", "epsilon"),
    ((-0.1, 0.1), (1.1, 0.1), (0.9, -0.1), (0.9, 1.1)),
)
def test_control_rejects_invalid_configuration(
    constructor: ControlConstructor,
    discount: float,
    epsilon: float,
) -> None:
    model = ActionValueNetwork(1, 2, hidden_sizes=())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError):
        constructor(model, optimizer, discount=discount, epsilon=epsilon)


def test_sarsa_requires_a_next_action_to_bootstrap() -> None:
    agent, _ = linear_control_agent(SemiGradientSARSA)
    assert isinstance(agent, SemiGradientSARSA)

    with pytest.raises(ValueError):
        agent.update([1.0], 0, 0.0, [2.0], None, terminated=False)


@pytest.mark.parametrize("constructor", (SemiGradientSARSA, SemiGradientQLearning))
@pytest.mark.parametrize("action", (-1, 2))
def test_control_rejects_an_invalid_action(
    constructor: ControlConstructor, action: int
) -> None:
    agent, _ = linear_control_agent(constructor)

    with pytest.raises(ValueError):
        if isinstance(agent, SemiGradientSARSA):
            agent.update([1.0], action, 0.0, [2.0], 0, terminated=False)
        else:
            agent.update([1.0], action, 0.0, [2.0], terminated=False)


@pytest.mark.parametrize("reward", (np.nan, np.inf, -np.inf))
def test_prediction_rejects_nonfinite_reward(reward: float) -> None:
    predictor, _ = linear_predictor()

    with pytest.raises(ValueError):
        predictor.update([1.0], reward, [2.0], terminated=False)


@pytest.mark.parametrize("constructor", (SemiGradientSARSA, SemiGradientQLearning))
@pytest.mark.parametrize("reward", (np.nan, np.inf, -np.inf))
def test_control_rejects_nonfinite_reward(
    constructor: ControlConstructor, reward: float
) -> None:
    agent, _ = linear_control_agent(constructor)

    with pytest.raises(ValueError):
        if isinstance(agent, SemiGradientSARSA):
            agent.update([1.0], 0, reward, [2.0], 1, terminated=False)
        else:
            agent.update([1.0], 0, reward, [2.0], terminated=False)
