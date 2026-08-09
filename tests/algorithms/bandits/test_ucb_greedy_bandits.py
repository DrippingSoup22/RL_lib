"""Tests for the UCB bandit agent."""

import numpy as np
import pytest

from rl_lib.algorithms.tabular.ucb_greedy_bandits import UCBGreedyBandits


def test_initial_values_and_counts() -> None:
    agent = UCBGreedyBandits(k=3, initial_value=5.0, seed=42)
    np.testing.assert_array_equal(agent.estimates, [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(agent.counts, [0, 0, 0])


def test_every_action_is_tried_once_before_ucb_calculation() -> None:
    agent = UCBGreedyBandits(k=5, seed=42)
    selected_actions = []
    with np.errstate(divide="raise", invalid="raise"):
        for _ in range(agent.k):
            action = agent.select_action()
            selected_actions.append(action)
            agent.update(action, reward=0.0)

    assert set(selected_actions) == set(range(agent.k))
    np.testing.assert_array_equal(agent.counts, np.ones(agent.k))


def test_update_computes_incremental_sample_average() -> None:
    agent = UCBGreedyBandits(k=3)

    for reward in (2.0, 4.0, 6.0):
        agent.update(action=1, reward=reward)

    assert agent.counts[1] == 3
    assert agent.estimates[1] == pytest.approx(4.0)
    np.testing.assert_array_equal(agent.counts[[0, 2]], [0, 0])
    np.testing.assert_array_equal(agent.estimates[[0, 2]], [0.0, 0.0])


def test_update_uses_constant_step_size_when_configured() -> None:
    agent = UCBGreedyBandits(k=3, step_size=0.5, seed=42)

    for reward in (2.0, 4.0, 6.0):
        agent.update(action=1, reward=reward)

    assert agent.counts[1] == 3
    assert agent.estimates[1] == pytest.approx(4.25)
    np.testing.assert_array_equal(agent.estimates[[0, 2]], [0.0, 0.0])


def test_constant_step_size_retains_part_of_initial_estimate() -> None:
    agent = UCBGreedyBandits(
        k=3,
        initial_value=10.0,
        step_size=0.25,
        seed=42,
    )

    agent.update(action=1, reward=2.0)

    assert agent.estimates[1] == pytest.approx(8.0)


def test_ucb_prefers_less_visited_action_when_bonus_is_large() -> None:
    agent = UCBGreedyBandits(k=2, c=1.0, seed=42)
    agent.estimates[:] = [1.0, 0.0]
    agent.counts[:] = [100, 1]

    assert agent.select_action() == 1


def test_zero_c_selects_highest_value_after_initialization() -> None:
    agent = UCBGreedyBandits(k=3, c=0.0, seed=42)
    agent.counts[:] = 1
    agent.estimates[:] = [1.0, 3.0, 2.0]

    assert agent.select_action() == 1


def test_ucb_randomly_breaks_ties() -> None:
    agent = UCBGreedyBandits(k=3, seed=42)
    agent.counts[:] = 1
    agent.estimates[:] = 2.0

    actions = {agent.select_action() for _ in range(30)}

    assert actions == {0, 1, 2}


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_number_of_actions_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k must be at least"):
        UCBGreedyBandits(k=k)


@pytest.mark.parametrize("exploration_constant", [-0.1, np.nan, np.inf])
def test_invalid_exploration_constant_is_rejected(
    exploration_constant: float,
) -> None:
    with pytest.raises(ValueError, match="c must be finite and non-negative"):
        UCBGreedyBandits(k=3, c=exploration_constant)


def test_exploration_constant_can_be_greater_than_one() -> None:
    agent = UCBGreedyBandits(k=3, c=2.0)

    assert agent.c == 2.0


@pytest.mark.parametrize("step_size", [-0.1, 0.0, 1.1, np.nan, np.inf])
def test_invalid_step_size_is_rejected(step_size: float) -> None:
    with pytest.raises(ValueError, match="step_size must be between"):
        UCBGreedyBandits(k=3, step_size=step_size)


@pytest.mark.parametrize("action", [-1, 3])
def test_update_rejects_invalid_action(action: int) -> None:
    agent = UCBGreedyBandits(k=3)

    with pytest.raises(ValueError, match="action must be between"):
        agent.update(action=action, reward=1.0)
