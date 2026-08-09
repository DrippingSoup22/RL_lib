"""Tests for episode data and return calculation."""

import numpy as np
import pytest

from rl_lib.data.episode import discounted_returns


def test_discounted_returns_include_future_rewards() -> None:
    returns = discounted_returns(rewards=[2.0, -1.0, 3.0], discount=0.5)

    np.testing.assert_allclose(returns, [2.25, 0.5, 3.0])


def test_discounted_returns_accept_numpy_array() -> None:
    rewards = np.array([2.0, -1.0, 3.0])

    returns = discounted_returns(rewards=rewards, discount=0.5)

    np.testing.assert_allclose(returns, [2.25, 0.5, 3.0])


def test_multidimensional_rewards_are_rejected() -> None:
    with pytest.raises(ValueError, match="rewards must be one-dimensional"):
        discounted_returns(rewards=[[1.0, 2.0]], discount=0.5)


def test_zero_discount_returns_immediate_rewards() -> None:
    returns = discounted_returns(rewards=[2.0, -1.0, 3.0], discount=0.0)

    np.testing.assert_array_equal(returns, [2.0, -1.0, 3.0])


def test_unit_discount_returns_undiscounted_suffix_sums() -> None:
    returns = discounted_returns(rewards=[2.0, -1.0, 3.0], discount=1.0)

    np.testing.assert_array_equal(returns, [4.0, 2.0, 3.0])


def test_single_reward_is_its_own_return() -> None:
    returns = discounted_returns(rewards=[7.5], discount=0.9)

    np.testing.assert_array_equal(returns, [7.5])


def test_empty_reward_sequence_produces_empty_returns() -> None:
    returns = discounted_returns(rewards=[], discount=0.9)

    assert returns.shape == (0,)


@pytest.mark.parametrize("discount", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_discount_is_rejected(discount: float) -> None:
    with pytest.raises(ValueError, match="discount must be finite and between"):
        discounted_returns(rewards=[1.0], discount=discount)
