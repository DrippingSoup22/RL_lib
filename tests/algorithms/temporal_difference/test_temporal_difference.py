import numpy as np
import pytest

from rl_lib.algorithms.temporal_difference import SARSA, QLearning, TDPrediction
from rl_lib.trajectories import EpisodeStep

ROLLOUT = (EpisodeStep(0, 0, 1.0), EpisodeStep(1, 0, 2.0))


def test_prediction_bootstraps_only_without_termination() -> None:
    predictor = TDPrediction(3, learning_rate=1.0, discount=0.5)
    predictor.V[2] = 8.0

    errors = predictor.update(ROLLOUT, final_state=2, terminated=False)

    assert errors == pytest.approx((4.0, 6.0))
    assert predictor.V == pytest.approx((4.0, 6.0, 8.0))

    predictor = TDPrediction(3, learning_rate=1.0, discount=0.5)
    predictor.V[2] = 100.0
    errors = predictor.update(ROLLOUT, final_state=2, terminated=True)
    assert errors == pytest.approx((2.0, 2.0))


def test_sarsa_bootstraps_from_the_selected_action_or_not_at_all() -> None:
    agent = SARSA(3, 2, learning_rate=1.0, discount=0.5, epsilon=0.0)
    agent.Q[2] = (4.0, 8.0)

    errors = agent.update(ROLLOUT, final_state=2, final_action=0, terminated=False)

    assert errors == pytest.approx((3.0, 4.0))
    assert agent.Q[0, 0] == pytest.approx(3.0)
    np.testing.assert_allclose(agent.policy.probabilities[0], (1.0, 0.0))

    # A terminated rollout needs no final action and ignores the final state.
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

    errors = agent.update(ROLLOUT, final_state=2, terminated=False)

    assert errors == pytest.approx((4.0, 6.0))
