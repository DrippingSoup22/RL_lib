import numpy as np
import pytest

from experiments.bandits.stationary_gaussian_bandit import StationaryGaussianBandit


def test_deterministic_rewards_equal_selected_action_value() -> None:
    environment = StationaryGaussianBandit(
        k=3,
        reward_std=0.0,
        seed=42,
    )
    action = 1

    rewards = [environment.step(action) for _ in range(3)]

    expected_reward = environment.action_values[action]
    assert rewards == pytest.approx([expected_reward] * 3)


def test_same_seed_creates_same_action_values() -> None:
    first = StationaryGaussianBandit(k=5, seed=42)
    second = StationaryGaussianBandit(k=5, seed=42)

    np.testing.assert_array_equal(
        first.action_values,
        second.action_values,
    )


def test_optimal_action_has_largest_action_value() -> None:
    environment = StationaryGaussianBandit(k=5, seed=42)

    expected_action = int(np.argmax(environment.action_values))

    assert environment.optimal_action == expected_action


def test_invalid_number_of_actions_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be at least"):
        StationaryGaussianBandit(k=0)


def test_negative_reward_standard_deviation_is_rejected() -> None:
    with pytest.raises(ValueError, match="reward_std must be non-negative"):
        StationaryGaussianBandit(k=3, reward_std=-0.1)


@pytest.mark.parametrize("action", [-1, 3])
def test_invalid_action_is_rejected(action: int) -> None:
    environment = StationaryGaussianBandit(k=3, seed=42)

    with pytest.raises(ValueError, match="action must be between"):
        environment.step(action)
