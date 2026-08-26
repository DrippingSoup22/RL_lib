import numpy as np
import pytest

from rl_lib.algorithms.monte_carlo import (
    EveryVisitMonteCarloControl,
    EveryVisitMonteCarloPrediction,
    FirstVisitMonteCarloControl,
    FirstVisitMonteCarloPrediction,
)
from rl_lib.data import Episode, EpisodeStep, discounted_returns
from rl_lib.policies import epsilon_soft_probabilities, policy_from_action_values


def repeated_state_episode() -> Episode[int]:
    return Episode(
        steps=(
            EpisodeStep(0, 0, 1.0),
            EpisodeStep(0, 0, 3.0),
        ),
        final_state=1,
        terminated=True,
        truncated=False,
    )


def test_discounted_returns_are_computed_backwards() -> None:
    np.testing.assert_allclose(discounted_returns([1, 2, 3], 0.5), [2.75, 3.5, 3])


def test_discounted_returns_can_bootstrap_after_the_final_reward() -> None:
    np.testing.assert_allclose(
        discounted_returns([1, 2], 0.5, bootstrap=4.0),
        [3.0, 4.0],
    )


def test_first_and_every_visit_prediction_count_different_samples() -> None:
    first = FirstVisitMonteCarloPrediction(2)
    every_visit = EveryVisitMonteCarloPrediction(2)
    episode = repeated_state_episode()
    first.update(episode)
    every_visit.update(episode)
    assert first.V[0] == pytest.approx(4.0)
    assert every_visit.V[0] == pytest.approx(3.5)
    assert first.visit_counts[0] == 1
    assert every_visit.visit_counts[0] == 2


@pytest.mark.parametrize(
    ("control_class", "expected_count"),
    [(FirstVisitMonteCarloControl, 1), (EveryVisitMonteCarloControl, 2)],
)
def test_control_updates_q_and_improves_policy(
    control_class: type, expected_count: int
) -> None:
    control = control_class(2, 2, epsilon=0.2, seed=0)
    control.update(repeated_state_episode())
    assert control.visit_counts[0, 0] == expected_count
    assert control.Q[0, 0] > control.Q[0, 1]
    np.testing.assert_allclose(control.policy.probabilities[0], [0.9, 0.1])


def test_epsilon_soft_policy_splits_greedy_ties() -> None:
    np.testing.assert_allclose(
        epsilon_soft_probabilities([2.0, 2.0, 0.0], epsilon=0.3),
        [0.45, 0.45, 0.1],
    )


def test_policy_from_action_values_can_be_greedy() -> None:
    policy = policy_from_action_values([[0.0, 1.0], [3.0, 2.0]], epsilon=0.0)
    np.testing.assert_allclose(policy.probabilities, [[0.0, 1.0], [1.0, 0.0]])
