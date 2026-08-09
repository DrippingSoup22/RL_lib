"""Tests for Monte Carlo stochastic policies and policy improvement."""

import numpy as np
import pytest

from rl_lib.algorithms.tabular.policies import (
    TabularPolicy,
    epsilon_soft_probabilities,
    policy_from_action_values,
)


def test_epsilon_soft_probabilities_favor_unique_greedy_action() -> None:
    probabilities = epsilon_soft_probabilities([1.0, 4.0, 2.0], epsilon=0.3)

    np.testing.assert_allclose(probabilities, [0.1, 0.8, 0.1])


def test_epsilon_soft_probabilities_split_probability_between_ties() -> None:
    probabilities = epsilon_soft_probabilities([1.0, 4.0, 4.0], epsilon=0.3)

    np.testing.assert_allclose(probabilities, [0.1, 0.45, 0.45])


def test_zero_epsilon_selects_only_greedy_actions() -> None:
    probabilities = epsilon_soft_probabilities([3.0, 1.0, 3.0], epsilon=0.0)

    np.testing.assert_allclose(probabilities, [0.5, 0.0, 0.5])


def test_unit_epsilon_produces_uniform_distribution() -> None:
    probabilities = epsilon_soft_probabilities([1.0, 8.0, -2.0], epsilon=1.0)

    np.testing.assert_allclose(probabilities, [1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize("action_values", [[], [[1.0, 2.0]]])
def test_invalid_action_value_shape_is_rejected(
    action_values: list[float] | list[list[float]],
) -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        epsilon_soft_probabilities(action_values, epsilon=0.1)


@pytest.mark.parametrize("action_values", [[1.0, np.nan], [1.0, np.inf]])
def test_nonfinite_action_values_are_rejected(action_values: list[float]) -> None:
    with pytest.raises(ValueError, match="action_values must be finite"):
        epsilon_soft_probabilities(action_values, epsilon=0.1)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_epsilon_is_rejected(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon must be finite and between"):
        epsilon_soft_probabilities([1.0, 2.0], epsilon=epsilon)


def test_policy_starts_uniform_in_every_state() -> None:
    policy = TabularPolicy(number_of_states=2, number_of_actions=3, seed=42)

    np.testing.assert_allclose(
        policy.probabilities,
        [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]],
    )


def test_policy_samples_only_action_with_probability_one() -> None:
    policy = TabularPolicy(number_of_states=2, number_of_actions=3, seed=42)
    policy.set_action_probabilities(state=1, probabilities=[0.0, 1.0, 0.0])

    actions = [policy.select_action(state=1) for _ in range(10)]

    assert actions == [1] * 10


def test_setting_one_state_does_not_change_other_states() -> None:
    policy = TabularPolicy(number_of_states=2, number_of_actions=2, seed=42)
    policy.set_action_probabilities(state=1, probabilities=[0.8, 0.2])

    np.testing.assert_allclose(policy.probabilities[0], [0.5, 0.5])
    np.testing.assert_allclose(policy.probabilities[1], [0.8, 0.2])


@pytest.mark.parametrize("state", [-1, 2])
def test_select_action_rejects_invalid_state(state: int) -> None:
    policy = TabularPolicy(number_of_states=2, number_of_actions=2)

    with pytest.raises(ValueError, match="state must be between"):
        policy.select_action(state)


@pytest.mark.parametrize("state", [-1, 2])
def test_setting_probabilities_rejects_invalid_state(state: int) -> None:
    policy = TabularPolicy(number_of_states=2, number_of_actions=2)

    with pytest.raises(ValueError, match="state must be between"):
        policy.set_action_probabilities(state, [0.5, 0.5])


@pytest.mark.parametrize(
    "probabilities, message",
    [
        ([1.0], "one value for every action"),
        ([0.5, np.nan], "probabilities must be finite"),
        ([-0.1, 1.1], "probabilities must be between"),
        ([0.2, 0.2], "probabilities must sum to 1"),
    ],
)
def test_invalid_probability_distribution_is_rejected(
    probabilities: list[float],
    message: str,
) -> None:
    policy = TabularPolicy(number_of_states=1, number_of_actions=2)

    with pytest.raises(ValueError, match=message):
        policy.set_action_probabilities(0, probabilities)


@pytest.mark.parametrize(
    "number_of_states, number_of_actions, message",
    [
        (0, 2, "number_of_states must be at least"),
        (2, 0, "number_of_actions must be at least"),
    ],
)
def test_invalid_space_size_is_rejected(
    number_of_states: int,
    number_of_actions: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TabularPolicy(number_of_states, number_of_actions)


def test_policy_from_action_values_builds_epsilon_soft_rows() -> None:
    policy = policy_from_action_values(
        [[4.0, 1.0], [2.0, 5.0]],
        epsilon=0.2,
        seed=42,
    )

    np.testing.assert_allclose(policy.probabilities, [[0.9, 0.1], [0.1, 0.9]])


def test_policy_from_action_values_builds_greedy_policy() -> None:
    policy = policy_from_action_values(
        [[4.0, 1.0], [2.0, 5.0]],
        epsilon=0.0,
    )

    np.testing.assert_allclose(policy.probabilities, [[1.0, 0.0], [0.0, 1.0]])


def test_policy_from_action_values_splits_greedy_ties() -> None:
    policy = policy_from_action_values([[3.0, 3.0, 1.0]], epsilon=0.0)

    np.testing.assert_allclose(policy.probabilities, [[0.5, 0.5, 0.0]])


def test_policy_from_action_values_copies_the_values() -> None:
    action_values = np.array([[3.0, 1.0]])
    policy = policy_from_action_values(action_values, epsilon=0.0)

    action_values[0] = [0.0, 4.0]

    np.testing.assert_allclose(policy.probabilities, [[1.0, 0.0]])


@pytest.mark.parametrize("action_values", [[1.0, 2.0], [], [[]]])
def test_policy_from_action_values_rejects_invalid_shape(
    action_values: list[float] | list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        policy_from_action_values(action_values, epsilon=0.1)


@pytest.mark.parametrize("action_values", [[[1.0, np.nan]], [[1.0, np.inf]]])
def test_policy_from_action_values_rejects_nonfinite_values(
    action_values: list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        policy_from_action_values(action_values, epsilon=0.1)
