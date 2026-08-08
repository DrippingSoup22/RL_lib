import numpy as np
import pytest

from experiments.bandits.nonstationary_gaussian_bandit import (
    NonstationaryGaussianBandit,
)


def test_action_values_start_at_initial_value() -> None:
    environment = NonstationaryGaussianBandit(
        k=4,
        initial_value=2.5,
        seed=42,
    )

    np.testing.assert_array_equal(environment.action_values, np.full(4, 2.5))


def test_reward_uses_selected_action_value_before_drift() -> None:
    environment = NonstationaryGaussianBandit(
        k=3,
        reward_std=0.0,
        drift_std=0.1,
        initial_value=2.0,
        seed=42,
    )

    reward = environment.step(1)

    assert reward == 2.0
    assert np.any(environment.action_values != 2.0)


def test_zero_drift_keeps_action_values_unchanged() -> None:
    environment = NonstationaryGaussianBandit(
        k=3,
        reward_std=0.0,
        drift_std=0.0,
        initial_value=1.0,
        seed=42,
    )
    initial_values = environment.action_values.copy()

    for action in range(environment.k):
        environment.step(action)

    np.testing.assert_array_equal(environment.action_values, initial_values)


def test_same_seed_creates_same_reward_and_drift_trajectory() -> None:
    first = NonstationaryGaussianBandit(k=3, seed=42)
    second = NonstationaryGaussianBandit(k=3, seed=42)

    for action in [0, 2, 1]:
        assert first.step(action) == second.step(action)
        np.testing.assert_array_equal(first.action_values, second.action_values)


def test_optimal_action_tracks_current_action_values() -> None:
    environment = NonstationaryGaussianBandit(k=3, seed=42)

    environment.action_values[:] = [1.0, 3.0, 2.0]
    assert environment.optimal_action == 1

    environment.action_values[:] = [4.0, 3.0, 2.0]
    assert environment.optimal_action == 0


def test_invalid_number_of_actions_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be at least"):
        NonstationaryGaussianBandit(k=0)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("reward_std", -0.1, "reward_std must be non-negative"),
        ("drift_std", -0.1, "drift_std must be non-negative"),
        ("initial_value", np.nan, "initial_value must be finite"),
        ("initial_value", np.inf, "initial_value must be finite"),
    ],
)
def test_invalid_parameters_are_rejected(
    parameter: str,
    value: float,
    message: str,
) -> None:
    arguments = {parameter: value}

    with pytest.raises(ValueError, match=message):
        NonstationaryGaussianBandit(k=3, **arguments)


@pytest.mark.parametrize("action", [-1, 3])
def test_invalid_action_is_rejected(action: int) -> None:
    environment = NonstationaryGaussianBandit(k=3, seed=42)

    with pytest.raises(ValueError, match="action must be between"):
        environment.step(action)
