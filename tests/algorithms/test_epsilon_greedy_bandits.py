import numpy as np
import pytest

from rl_lib.algorithms.tabular.epsilon_greedy_bandits import EpsilonGreedyBandits


def test_initial_estimates_and_counts() -> None:
    agent = EpsilonGreedyBandits(k=3, initial_value=5.0, seed=42)

    np.testing.assert_array_equal(agent.estimates, [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(agent.counts, [0, 0, 0])


def test_optimistic_values_try_each_action_before_repeating() -> None:
    agent = EpsilonGreedyBandits(
        k=5,
        epsilon=0.0,
        initial_value=10.0,
        seed=42,
    )

    selected_actions = []
    for _ in range(agent.k):
        action = agent.select_action()
        selected_actions.append(action)
        agent.update(action, reward=0.0)

    assert set(selected_actions) == set(range(agent.k))


def test_update_computes_incremental_sample_average() -> None:
    agent = EpsilonGreedyBandits(k=3, seed=42)

    for reward in (2.0, 4.0, 6.0):
        agent.update(action=1, reward=reward)

    assert agent.counts[1] == 3
    assert agent.estimates[1] == pytest.approx(4.0)
    np.testing.assert_array_equal(agent.estimates[[0, 2]], [0.0, 0.0])


def test_update_uses_constant_step_size_when_configured() -> None:
    agent = EpsilonGreedyBandits(k=3, step_size=0.5, seed=42)

    for reward in (2.0, 4.0, 6.0):
        agent.update(action=1, reward=reward)

    assert agent.counts[1] == 3
    assert agent.estimates[1] == pytest.approx(4.25)
    np.testing.assert_array_equal(agent.estimates[[0, 2]], [0.0, 0.0])


def test_constant_step_size_retains_part_of_initial_estimate() -> None:
    agent = EpsilonGreedyBandits(
        k=3,
        initial_value=10.0,
        step_size=0.25,
        seed=42,
    )

    agent.update(action=1, reward=2.0)

    assert agent.estimates[1] == pytest.approx(8.0)


def test_greedy_policy_selects_unique_best_action() -> None:
    agent = EpsilonGreedyBandits(k=3, epsilon=0.0, seed=42)
    agent.estimates[:] = [1.0, 3.0, 2.0]

    actions = [agent.select_action() for _ in range(10)]

    assert actions == [1] * 10


def test_greedy_policy_randomly_breaks_ties() -> None:
    agent = EpsilonGreedyBandits(k=3, epsilon=0.0, seed=42)
    agent.estimates[:] = [1.0, 3.0, 3.0]

    actions = {agent.select_action() for _ in range(20)}

    assert actions == {1, 2}


def test_full_exploration_samples_different_valid_actions() -> None:
    agent = EpsilonGreedyBandits(k=3, epsilon=1.0, seed=42)

    actions = [agent.select_action() for _ in range(30)]

    assert set(actions) == {0, 1, 2}


def test_invalid_number_of_actions_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be at least"):
        EpsilonGreedyBandits(k=0)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_invalid_epsilon_is_rejected(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon must be between"):
        EpsilonGreedyBandits(k=3, epsilon=epsilon)


@pytest.mark.parametrize("step_size", [-0.1, 0.0, 1.1, np.nan, np.inf])
def test_invalid_step_size_is_rejected(step_size: float) -> None:
    with pytest.raises(ValueError, match="step_size must be between"):
        EpsilonGreedyBandits(k=3, step_size=step_size)


@pytest.mark.parametrize("action", [-1, 3])
def test_update_rejects_invalid_action(action: int) -> None:
    agent = EpsilonGreedyBandits(k=3)

    with pytest.raises(ValueError, match="action must be between"):
        agent.update(action=action, reward=1.0)
