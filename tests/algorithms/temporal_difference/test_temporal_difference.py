import numpy as np
import pytest

from rl_lib.algorithms.temporal_difference import (
    SARSA,
    Q_learning,
    TDZeroPrediction,
)


def test_td_zero_bootstraps_from_next_state() -> None:
    predictor = TDZeroPrediction(2, learning_rate=0.5, discount=0.9)
    predictor.V[1] = 4.0

    predictor.update(state=0, reward=1.0, next_state=1, terminated=False)

    assert predictor.V[0] == pytest.approx(2.3)


def test_td_zero_does_not_bootstrap_after_termination() -> None:
    predictor = TDZeroPrediction(2, learning_rate=0.5, discount=0.9)
    predictor.V[:] = (2.0, 100.0)

    predictor.update(state=0, reward=4.0, next_state=1, terminated=True)

    assert predictor.V[0] == pytest.approx(3.0)


def test_sarsa_uses_the_selected_next_action() -> None:
    agent = SARSA(2, 2, learning_rate=0.5, discount=0.9, epsilon=0.2, seed=0)
    agent.Q[1] = (2.0, 4.0)

    agent.update(
        state=0,
        action=0,
        reward=1.0,
        next_state=1,
        next_action=1,
        terminated=False,
    )

    assert agent.Q[0, 0] == pytest.approx(2.3)
    assert agent.Q[0, 1] == 0.0
    np.testing.assert_allclose(agent.policy.probabilities[0], (0.9, 0.1))


def test_sarsa_terminal_transition_does_not_require_a_next_action() -> None:
    agent = SARSA(2, 2, learning_rate=1.0)
    agent.Q[1, 1] = 100.0

    agent.update(
        state=0,
        action=0,
        reward=3.0,
        next_state=1,
        next_action=None,
        terminated=True,
    )

    assert agent.Q[0, 0] == pytest.approx(3.0)


def test_q_learning_uses_the_largest_next_action_value() -> None:
    agent = Q_learning(
        2,
        2,
        learning_rate=0.5,
        discount=0.9,
        epsilon=0.2,
        seed=0,
    )
    agent.Q[1] = (2.0, 4.0)

    agent.update(
        state=0,
        action=0,
        reward=1.0,
        next_state=1,
        terminated=False,
    )

    assert agent.Q[0, 0] == pytest.approx(2.3)
    assert agent.Q[0, 1] == 0.0
    np.testing.assert_allclose(agent.policy.probabilities[0], (0.9, 0.1))
