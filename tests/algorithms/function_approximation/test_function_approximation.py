from collections.abc import Callable

import numpy as np
import pytest
import torch

from rl_lib.algorithms.function_approximation import (
    SemiGradientQLearning,
    SemiGradientSARSA,
    SemiGradientTDPrediction,
)
from rl_lib.data import EpisodeStep
from rl_lib.models import ActionValueNetwork, StateValueNetwork

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


def test_prediction_uses_bootstrapped_rollout_returns() -> None:
    predictor, _ = linear_predictor()

    errors = predictor.update(
        rollout(), np.array([3.0], dtype=np.float32), terminated=False
    )

    assert errors == pytest.approx((1.5, 1.0))


def test_prediction_does_not_bootstrap_after_termination() -> None:
    predictor, _ = linear_predictor()

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


def test_sarsa_requires_an_action_for_nonterminal_bootstrap() -> None:
    agent, _ = linear_control_agent(SemiGradientSARSA)
    with pytest.raises(ValueError, match="Final action"):
        agent.update(
            rollout(),
            np.array([3.0], dtype=np.float32),
            final_action=None,
            terminated=False,
        )


@pytest.mark.parametrize(
    "constructor", (SemiGradientTDPrediction, SemiGradientSARSA, SemiGradientQLearning)
)
def test_algorithms_reject_an_empty_rollout(constructor: type) -> None:
    model: StateValueNetwork | ActionValueNetwork
    if constructor is SemiGradientTDPrediction:
        model = StateValueNetwork(1, hidden_sizes=())
    else:
        model = ActionValueNetwork(1, 2, hidden_sizes=())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    agent = constructor(model, optimizer)

    with pytest.raises(ValueError, match="empty"):
        if isinstance(agent, SemiGradientSARSA):
            agent.update((), [0.0], 0, terminated=False)
        else:
            agent.update((), [0.0], terminated=False)


@pytest.mark.parametrize("discount", (-0.1, 1.1, np.nan, np.inf))
def test_prediction_rejects_invalid_discount(discount: float) -> None:
    model = StateValueNetwork(1, hidden_sizes=())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError):
        SemiGradientTDPrediction(model, optimizer, discount)


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


@pytest.mark.parametrize("constructor", (SemiGradientSARSA, SemiGradientQLearning))
@pytest.mark.parametrize("action", (-1, 2))
def test_control_rejects_an_invalid_rollout_action(
    constructor: ControlConstructor,
    action: int,
) -> None:
    agent, _ = linear_control_agent(constructor)
    steps = (EpisodeStep(np.array([1.0], dtype=np.float32), action, 0.0),)
    with pytest.raises(ValueError, match="Actions"):
        if isinstance(agent, SemiGradientSARSA):
            agent.update(steps, [2.0], 0, terminated=False)
        else:
            agent.update(steps, [2.0], terminated=False)


@pytest.mark.parametrize("reward", (np.nan, np.inf, -np.inf))
def test_prediction_rejects_a_nonfinite_rollout_reward(reward: float) -> None:
    predictor, _ = linear_predictor()
    steps = (EpisodeStep(np.array([1.0], dtype=np.float32), 0, reward),)
    with pytest.raises(ValueError, match="finite"):
        predictor.update(steps, [2.0], terminated=False)


@pytest.mark.parametrize("constructor", (SemiGradientSARSA, SemiGradientQLearning))
def test_control_rejects_an_invalid_observation_shape(
    constructor: ControlConstructor,
) -> None:
    agent, _ = linear_control_agent(constructor)
    steps = (EpisodeStep(np.array([1.0, 2.0], dtype=np.float32), 0, 0.0),)
    with pytest.raises(ValueError, match="input size"):
        if isinstance(agent, SemiGradientSARSA):
            agent.update(steps, [2.0], 0, terminated=False)
        else:
            agent.update(steps, [2.0], terminated=False)
