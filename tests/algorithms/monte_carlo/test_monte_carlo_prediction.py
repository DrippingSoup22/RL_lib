"""Tests for Monte Carlo prediction visit modes."""

import numpy as np
import pytest

from rl_lib.algorithms.tabular.monte_carlo_prediction import (
    EveryVisitMonteCarloPrediction,
    FirstVisitMonteCarloPrediction,
)
from rl_lib.data.episode import Episode, EpisodeStep


def make_episode(
    steps: list[EpisodeStep],
    *,
    final_state: int = 2,
    terminated: bool = True,
    truncated: bool = False,
) -> Episode:
    return Episode(
        steps=tuple(steps),
        final_state=final_state,
        terminated=terminated,
        truncated=truncated,
    )


@pytest.mark.parametrize(
    "prediction_class",
    [FirstVisitMonteCarloPrediction, EveryVisitMonteCarloPrediction],
)
def test_initial_values_and_visit_counts_are_zero(prediction_class: type) -> None:
    prediction = prediction_class(number_of_states=3)

    np.testing.assert_array_equal(prediction.values, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(prediction.visit_counts, [0, 0, 0])


def test_successful_episode_updates_each_visited_state() -> None:
    prediction = FirstVisitMonteCarloPrediction(number_of_states=4)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=1, reward=1.0),
        ],
        final_state=2,
    )

    prediction.update(episode)

    np.testing.assert_array_equal(prediction.values, [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(prediction.visit_counts, [1, 1, 0, 0])


def test_only_first_visit_of_repeated_state_is_used() -> None:
    prediction = FirstVisitMonteCarloPrediction(number_of_states=3)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=1, reward=1.0),
        ]
    )

    prediction.update(episode)

    np.testing.assert_allclose(prediction.values, [0.8, 0.8, 0.0])
    np.testing.assert_array_equal(prediction.visit_counts, [1, 1, 0])


def test_every_visit_of_repeated_state_is_used() -> None:
    prediction = EveryVisitMonteCarloPrediction(number_of_states=3)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=1, reward=1.0),
        ]
    )

    prediction.update(episode)

    np.testing.assert_allclose(prediction.values, [0.8, 0.9, 0.0])
    np.testing.assert_array_equal(prediction.visit_counts, [1, 3, 0])


@pytest.mark.parametrize(
    "prediction_class",
    [FirstVisitMonteCarloPrediction, EveryVisitMonteCarloPrediction],
)
def test_returns_are_averaged_across_episodes(prediction_class: type) -> None:
    prediction = prediction_class(number_of_states=2)
    first_episode = make_episode(
        [EpisodeStep(state=0, action=0, reward=0.8)],
        final_state=1,
    )
    second_episode = make_episode(
        [EpisodeStep(state=0, action=0, reward=1.0)],
        final_state=1,
    )

    prediction.update(first_episode)
    prediction.update(second_episode)

    assert prediction.values[0] == pytest.approx(0.9)
    assert prediction.visit_counts[0] == 2


@pytest.mark.parametrize(
    "prediction_class",
    [FirstVisitMonteCarloPrediction, EveryVisitMonteCarloPrediction],
)
def test_discount_is_applied_to_future_rewards(prediction_class: type) -> None:
    prediction = prediction_class(
        number_of_states=3,
        discount=0.5,
    )
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=1, reward=1.0),
        ]
    )

    prediction.update(episode)

    np.testing.assert_allclose(prediction.values, [0.5, 1.0, 0.0])


def test_truncated_episode_uses_all_collected_rewards() -> None:
    prediction = FirstVisitMonteCarloPrediction(number_of_states=2)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=0, reward=-0.1),
        ],
        final_state=1,
        terminated=False,
        truncated=True,
    )

    prediction.update(episode)

    np.testing.assert_allclose(prediction.values, [-0.2, -0.2])
    np.testing.assert_array_equal(prediction.visit_counts, [1, 1])


def test_every_visit_uses_all_repeated_visits_in_truncated_episode() -> None:
    prediction = EveryVisitMonteCarloPrediction(number_of_states=2)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=0, reward=-0.1),
        ],
        final_state=1,
        terminated=False,
        truncated=True,
    )

    prediction.update(episode)

    np.testing.assert_allclose(prediction.values, [-0.2, -0.15])
    np.testing.assert_array_equal(prediction.visit_counts, [1, 2])


@pytest.mark.parametrize(
    "prediction_class",
    [FirstVisitMonteCarloPrediction, EveryVisitMonteCarloPrediction],
)
@pytest.mark.parametrize("number_of_states", [0, -1])
def test_nonpositive_number_of_states_is_rejected(
    prediction_class: type,
    number_of_states: int,
) -> None:
    with pytest.raises(ValueError, match="Number of state must be positive"):
        prediction_class(number_of_states)


@pytest.mark.parametrize(
    "prediction_class",
    [FirstVisitMonteCarloPrediction, EveryVisitMonteCarloPrediction],
)
@pytest.mark.parametrize("discount", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_discount_is_rejected(
    prediction_class: type,
    discount: float,
) -> None:
    with pytest.raises(ValueError, match="Discount must be finite and between"):
        prediction_class(number_of_states=2, discount=discount)


@pytest.mark.parametrize(
    "prediction_class",
    [FirstVisitMonteCarloPrediction, EveryVisitMonteCarloPrediction],
)
@pytest.mark.parametrize("state", [-1, 2])
def test_invalid_episode_state_is_rejected(
    prediction_class: type,
    state: int,
) -> None:
    prediction = prediction_class(number_of_states=2)
    episode = make_episode(
        [EpisodeStep(state=state, action=0, reward=1.0)],
        final_state=1,
    )

    with pytest.raises(ValueError, match="State value not valid"):
        prediction.update(episode)
