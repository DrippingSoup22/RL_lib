"""Tests for first-visit and every-visit on-policy Monte Carlo control."""

import numpy as np
import pytest

from rl_lib.algorithms.tabular.monte_carlo_control import (
    EveryVisitMonteCarloControl,
    FirstVisitMonteCarloControl,
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
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
def test_initial_action_values_counts_and_policy(control_class: type) -> None:
    control = control_class(
        number_of_states=3,
        number_of_actions=2,
        seed=42,
    )

    np.testing.assert_array_equal(control.action_values, np.zeros((3, 2)))
    np.testing.assert_array_equal(control.visit_counts, np.zeros((3, 2)))
    np.testing.assert_allclose(control.policy.probabilities, np.full((3, 2), 0.5))


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
def test_select_action_uses_owned_policy(control_class: type) -> None:
    control = control_class(2, 2, seed=42)
    control.policy.set_action_probabilities(0, [0.0, 1.0])

    actions = [control.select_action(0) for _ in range(10)]

    assert actions == [1] * 10


def test_first_visit_updates_action_values_and_improves_policy() -> None:
    control = FirstVisitMonteCarloControl(3, 2, epsilon=0.1)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=1, reward=1.0),
        ]
    )

    control.update(episode)

    np.testing.assert_allclose(
        control.action_values,
        [[0.8, 0.0], [0.8, 1.0], [0.0, 0.0]],
    )
    np.testing.assert_array_equal(
        control.visit_counts,
        [[1, 0], [1, 1], [0, 0]],
    )
    np.testing.assert_allclose(
        control.policy.probabilities,
        [[0.95, 0.05], [0.05, 0.95], [0.5, 0.5]],
    )


def test_every_visit_updates_repeated_state_action_pair() -> None:
    control = EveryVisitMonteCarloControl(3, 2, epsilon=0.1)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=0, reward=-0.1),
            EpisodeStep(state=1, action=1, reward=1.0),
        ]
    )

    control.update(episode)

    np.testing.assert_allclose(
        control.action_values,
        [[0.8, 0.0], [0.85, 1.0], [0.0, 0.0]],
    )
    np.testing.assert_array_equal(
        control.visit_counts,
        [[1, 0], [2, 1], [0, 0]],
    )
    np.testing.assert_allclose(
        control.policy.probabilities,
        [[0.95, 0.05], [0.05, 0.95], [0.5, 0.5]],
    )


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
def test_different_actions_in_same_state_are_distinct_visits(
    control_class: type,
) -> None:
    control = control_class(2, 2, epsilon=0.2)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=0, action=1, reward=1.0),
        ],
        final_state=1,
    )

    control.update(episode)

    np.testing.assert_allclose(control.action_values[0], [1.0, 1.0])
    np.testing.assert_array_equal(control.visit_counts[0], [1, 1])
    np.testing.assert_allclose(control.policy.probabilities[0], [0.5, 0.5])


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
def test_action_values_are_averaged_across_episodes(control_class: type) -> None:
    control = control_class(2, 2, epsilon=0.1)
    first_episode = make_episode(
        [EpisodeStep(state=0, action=0, reward=0.8)],
        final_state=1,
    )
    second_episode = make_episode(
        [EpisodeStep(state=0, action=0, reward=1.0)],
        final_state=1,
    )

    control.update(first_episode)
    control.update(second_episode)

    assert control.action_values[0, 0] == pytest.approx(0.9)
    assert control.visit_counts[0, 0] == 2
    np.testing.assert_allclose(control.policy.probabilities[0], [0.95, 0.05])


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
def test_discount_is_applied_to_future_rewards(control_class: type) -> None:
    control = control_class(3, 2, epsilon=0.1, discount=0.5)
    episode = make_episode(
        [
            EpisodeStep(state=0, action=0, reward=0.0),
            EpisodeStep(state=1, action=1, reward=1.0),
        ]
    )

    control.update(episode)

    assert control.action_values[0, 0] == pytest.approx(0.5)
    assert control.action_values[1, 1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
@pytest.mark.parametrize("number_of_states", [0, -1])
def test_nonpositive_number_of_states_is_rejected(
    control_class: type,
    number_of_states: int,
) -> None:
    with pytest.raises(ValueError, match="number_of_states must be at least"):
        control_class(number_of_states, number_of_actions=2)


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
@pytest.mark.parametrize("number_of_actions", [0, -1])
def test_nonpositive_number_of_actions_is_rejected(
    control_class: type,
    number_of_actions: int,
) -> None:
    with pytest.raises(ValueError, match="number_of_actions must be at least"):
        control_class(
            number_of_states=2,
            number_of_actions=number_of_actions,
        )


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
@pytest.mark.parametrize("epsilon", [0.0, -0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_epsilon_is_rejected(control_class: type, epsilon: float) -> None:
    with pytest.raises(ValueError, match="Epsilon must be between"):
        control_class(2, 2, epsilon=epsilon)


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
@pytest.mark.parametrize("discount", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_discount_is_rejected(control_class: type, discount: float) -> None:
    with pytest.raises(ValueError, match="Discount must be between"):
        control_class(2, 2, discount=discount)


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
@pytest.mark.parametrize("state", [-1, 2])
def test_invalid_episode_state_is_rejected(
    control_class: type,
    state: int,
) -> None:
    control = control_class(2, 2)
    episode = make_episode([EpisodeStep(state=state, action=0, reward=1.0)])

    with pytest.raises(ValueError, match="Invalid state"):
        control.update(episode)


@pytest.mark.parametrize(
    "control_class",
    [FirstVisitMonteCarloControl, EveryVisitMonteCarloControl],
)
@pytest.mark.parametrize("action", [-1, 2])
def test_invalid_episode_action_is_rejected(
    control_class: type,
    action: int,
) -> None:
    control = control_class(2, 2)
    episode = make_episode([EpisodeStep(state=0, action=action, reward=1.0)])

    with pytest.raises(ValueError, match="Invalid action"):
        control.update(episode)
