import numpy as np
import pytest

from rl_lib.algorithms.temporal_difference import SARSA, QLearning, TDPrediction
from rl_lib.data import EpisodeStep


def test_prediction_uses_bootstrapped_rollout_returns() -> None:
    predictor = TDPrediction(3, learning_rate=1.0, discount=0.5)
    predictor.V[2] = 8.0

    errors = predictor.update(
        (EpisodeStep(0, 0, 1.0), EpisodeStep(1, 0, 2.0)),
        final_state=2,
        terminated=False,
    )

    assert errors == pytest.approx((4.0, 6.0))
    assert predictor.V == pytest.approx((4.0, 6.0, 8.0))


def test_prediction_does_not_bootstrap_after_termination() -> None:
    predictor = TDPrediction(3, learning_rate=1.0, discount=0.5)
    predictor.V[2] = 100.0

    errors = predictor.update(
        (EpisodeStep(0, 0, 1.0), EpisodeStep(1, 0, 2.0)),
        final_state=2,
        terminated=True,
    )

    assert errors == pytest.approx((2.0, 2.0))


def test_sarsa_uses_the_selected_bootstrap_action() -> None:
    agent = SARSA(3, 2, learning_rate=1.0, discount=0.5, epsilon=0.0)
    agent.Q[2] = (4.0, 8.0)

    errors = agent.update(
        (EpisodeStep(0, 0, 1.0), EpisodeStep(1, 0, 2.0)),
        final_state=2,
        final_action=0,
        terminated=False,
    )

    assert errors == pytest.approx((3.0, 4.0))
    assert agent.Q[0, 0] == pytest.approx(3.0)
    np.testing.assert_allclose(agent.policy.probabilities[0], (1.0, 0.0))


def test_sarsa_terminal_rollout_does_not_require_a_final_action() -> None:
    agent = SARSA(3, 2, learning_rate=1.0, discount=0.5)
    agent.Q[2, 1] = 100.0

    agent.update(
        (EpisodeStep(0, 0, 1.0), EpisodeStep(1, 1, 2.0)),
        final_state=2,
        final_action=None,
        terminated=True,
    )

    assert agent.Q[0, 0] == pytest.approx(2.0)
    assert agent.Q[1, 1] == pytest.approx(2.0)


def test_q_learning_uses_the_greedy_bootstrap_value() -> None:
    agent = QLearning(3, 2, learning_rate=1.0, discount=0.5, epsilon=0.0)
    agent.Q[2] = (4.0, 8.0)

    errors = agent.update(
        (EpisodeStep(0, 0, 1.0), EpisodeStep(1, 0, 2.0)),
        final_state=2,
        terminated=False,
    )

    assert errors == pytest.approx((4.0, 6.0))


@pytest.mark.parametrize("constructor", (TDPrediction, SARSA, QLearning))
def test_td_algorithms_reject_an_empty_rollout(constructor: type) -> None:
    agent = constructor(2) if constructor is TDPrediction else constructor(2, 2)
    with pytest.raises(ValueError, match="empty"):
        if isinstance(agent, SARSA):
            agent.update((), 0, 0, terminated=False)
        else:
            agent.update((), 0, terminated=False)


def test_sarsa_requires_an_action_for_nonterminal_bootstrap() -> None:
    agent = SARSA(2, 2)
    with pytest.raises(ValueError, match="Final action"):
        agent.update(
            (EpisodeStep(0, 0, 1.0),),
            final_state=1,
            final_action=None,
            terminated=False,
        )


@pytest.mark.parametrize("reward", (np.nan, np.inf, -np.inf))
def test_tabular_prediction_rejects_a_nonfinite_reward(reward: float) -> None:
    predictor = TDPrediction(2)
    with pytest.raises(ValueError, match="finite"):
        predictor.update((EpisodeStep(0, 0, reward),), final_state=1, terminated=False)


@pytest.mark.parametrize("constructor", (SARSA, QLearning))
@pytest.mark.parametrize("action", (-1, 2))
def test_tabular_control_rejects_an_invalid_action(
    constructor: type[SARSA] | type[QLearning],
    action: int,
) -> None:
    agent = constructor(2, 2)
    steps = (EpisodeStep(0, action, 0.0),)
    with pytest.raises(ValueError, match="Action"):
        if isinstance(agent, SARSA):
            agent.update(steps, 1, 0, terminated=False)
        else:
            agent.update(steps, 1, terminated=False)
