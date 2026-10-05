from collections.abc import Callable

import numpy as np
import pytest
import torch

from rl_lib.algorithms.function_approximation import (
    SemiGradientQLearning,
    SemiGradientSARSA,
    SemiGradientTDPrediction,
)
from rl_lib.networks import ActionValueNetwork, StateValueNetwork
from rl_lib.trajectories import EpisodeStep

ControlAgent = SemiGradientSARSA | SemiGradientQLearning
ControlConstructor = Callable[..., ControlAgent]


def linear_predictor(
    *, discount: float = 0.5, learning_rate: float = 0.0
) -> tuple[SemiGradientTDPrediction, torch.nn.Linear]:
    model = StateValueNetwork(1, hidden_sizes=())
    layer = model.network[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.fill_(2.0)
        layer.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return SemiGradientTDPrediction(model, optimizer, discount), layer


def linear_control_agent(
    constructor: ControlConstructor,
    *,
    discount: float = 0.5,
    epsilon: float = 0.0,
    learning_rate: float = 0.0,
) -> tuple[ControlAgent, torch.nn.Linear]:
    model = ActionValueNetwork(1, 2, hidden_sizes=())
    layer = model.network[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0], [2.0]]))
        layer.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return (
        constructor(model, optimizer, discount=discount, epsilon=epsilon, seed=0),
        layer,
    )


def rollout() -> tuple[EpisodeStep[np.ndarray], ...]:
    return (
        EpisodeStep(np.array([1.0], dtype=np.float32), 0, 1.0),
        EpisodeStep(np.array([2.0], dtype=np.float32), 1, 2.0),
    )


def test_prediction_bootstraps_only_without_termination() -> None:
    predictor, _ = linear_predictor()

    errors = predictor.update(
        rollout(), np.array([3.0], dtype=np.float32), terminated=False
    )
    assert errors == pytest.approx((1.5, 1.0))

    errors = predictor.update(
        rollout(), np.array([100.0], dtype=np.float32), terminated=True
    )
    assert errors == pytest.approx((0.0, -2.0))


def test_prediction_takes_one_mean_loss_update_for_the_rollout() -> None:
    predictor, layer = linear_predictor(learning_rate=0.1)

    predictor.update(rollout(), np.array([3.0], dtype=np.float32), terminated=False)

    torch.testing.assert_close(layer.weight, torch.tensor([[2.175]]))
    torch.testing.assert_close(layer.bias, torch.tensor([0.125]))


def test_sarsa_uses_the_selected_bootstrap_action() -> None:
    agent, _ = linear_control_agent(SemiGradientSARSA)

    errors = agent.update(
        rollout(),
        np.array([3.0], dtype=np.float32),
        final_action=0,
        terminated=False,
    )

    assert errors == pytest.approx((1.75, -0.5))


def test_q_learning_uses_the_greedy_bootstrap_value() -> None:
    agent, _ = linear_control_agent(SemiGradientQLearning)

    errors = agent.update(
        rollout(), np.array([3.0], dtype=np.float32), terminated=False
    )

    assert errors == pytest.approx((2.5, 1.0))


@pytest.mark.parametrize("constructor", (SemiGradientSARSA, SemiGradientQLearning))
def test_control_selects_an_epsilon_greedy_action(
    constructor: ControlConstructor,
) -> None:
    agent, _ = linear_control_agent(constructor)
    assert agent.select_action([1.0]) == 1


@pytest.mark.parametrize(
    ("constructor", "invalid_settings"),
    (
        (SemiGradientTDPrediction, ({"discount": 1.1}, {"discount": np.nan})),
        (SemiGradientSARSA, ({"discount": -0.1}, {"epsilon": 1.1})),
        (SemiGradientQLearning, ({"discount": 1.1}, {"epsilon": -0.1})),
    ),
)
def test_constructors_reject_invalid_settings(constructor, invalid_settings) -> None:
    if constructor is SemiGradientTDPrediction:
        model = StateValueNetwork(1, hidden_sizes=())
    else:
        model = ActionValueNetwork(1, 2, hidden_sizes=())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for settings in invalid_settings:
        with pytest.raises(ValueError):
            constructor(model, optimizer, **settings)
