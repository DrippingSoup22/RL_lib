import numpy as np
import pytest

from rl_lib.algorithms.policy_gradient import generalized_advantage_estimates


def test_gae_ignores_the_final_value_after_true_termination() -> None:
    advantages, return_targets = generalized_advantage_estimates(
        [1.0, 1.0],
        [0.5, 0.25],
        final_value=100.0,
        terminated=True,
        discount=0.9,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(advantages, [1.4, 0.75])
    np.testing.assert_allclose(return_targets, [1.9, 1.0])


def test_gae_bootstraps_after_a_nonterminal_rollout_boundary() -> None:
    advantages, return_targets = generalized_advantage_estimates(
        [1.0, 1.0],
        [0.5, 0.25],
        final_value=2.0,
        terminated=False,
        discount=0.9,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(advantages, [3.02, 2.55])
    np.testing.assert_allclose(return_targets, [3.52, 2.8])


def test_zero_lambda_reduces_gae_to_one_step_td_residuals() -> None:
    advantages, return_targets = generalized_advantage_estimates(
        [1.0, 2.0],
        [0.5, 0.25],
        final_value=2.0,
        terminated=False,
        discount=0.9,
        gae_lambda=0.0,
    )

    np.testing.assert_allclose(advantages, [0.725, 3.55])
    np.testing.assert_allclose(return_targets, [1.225, 3.8])


@pytest.mark.parametrize(
    ("rewards", "values", "message"),
    (
        (1.0, [0.0], "one-dimensional"),
        ([[1.0]], [0.0], "one-dimensional"),
        ([], [], "non empty"),
        ([1.0, 2.0], [0.0], "same length"),
        ([1.0, np.nan], [0.0, 0.0], "rewards"),
        ([1.0, 2.0], [0.0, np.inf], "values"),
    ),
)
def test_gae_rejects_invalid_trajectories(
    rewards,
    values,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generalized_advantage_estimates(
            rewards,
            values,
            final_value=0.0,
            terminated=True,
        )


@pytest.mark.parametrize("final_value", (np.nan, np.inf, -np.inf))
def test_gae_rejects_a_nonfinite_final_value(final_value: float) -> None:
    with pytest.raises(ValueError, match="Final value"):
        generalized_advantage_estimates(
            [1.0],
            [0.0],
            final_value,
            terminated=False,
        )


@pytest.mark.parametrize("discount", (-0.1, 1.1, np.nan, np.inf, -np.inf))
def test_gae_rejects_an_invalid_discount(discount: float) -> None:
    with pytest.raises(ValueError, match="Discount"):
        generalized_advantage_estimates(
            [1.0],
            [0.0],
            0.0,
            terminated=True,
            discount=discount,
        )


@pytest.mark.parametrize("gae_lambda", (-0.1, 1.1, np.nan, np.inf, -np.inf))
def test_gae_rejects_an_invalid_lambda(gae_lambda: float) -> None:
    with pytest.raises(ValueError, match="Gae lambda"):
        generalized_advantage_estimates(
            [1.0],
            [0.0],
            0.0,
            terminated=True,
            gae_lambda=gae_lambda,
        )
