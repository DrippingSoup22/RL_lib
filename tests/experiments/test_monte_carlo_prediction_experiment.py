"""Tests for the Monte Carlo prediction comparison experiment."""

import numpy as np
import pytest

from experiments.monte_carlo.prediction import (
    EPISODE_FIELDS,
    TRACE_FIELDS,
    compare_prediction_methods,
    render_report,
    true_looping_mdp_values,
)

ACTION_VALUES = np.array(
    [
        [1.0, -1.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
)


def test_true_values_match_looping_mdp_bellman_equations() -> None:
    policy_probabilities = np.array(
        [
            [0.75, 0.25],
            [0.25, 0.75],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    values = true_looping_mdp_values(policy_probabilities, discount=1.0)

    np.testing.assert_allclose(values, [0.475, 29 / 30, 0.0, 0.0])


def test_comparison_updates_both_methods_from_one_shared_episode() -> None:
    result = compare_prediction_methods(
        run_id="test-run",
        action_values=ACTION_VALUES,
        episodes=1,
        epsilon=0.5,
        discount=1.0,
        max_episode_steps=10,
        seed=3,
    )

    assert len(result.episode_records) == 1
    assert len(result.trace_records) == 2
    assert tuple(result.episode_records[0]) == EPISODE_FIELDS
    assert all(tuple(record) == TRACE_FIELDS for record in result.trace_records)
    np.testing.assert_allclose(
        result.policy_probabilities,
        [[0.75, 0.25], [0.25, 0.75], [0.5, 0.5], [0.5, 0.5]],
    )
    np.testing.assert_allclose(
        result.final_values["first_visit_mc_prediction"],
        [0.9, 0.9, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        result.final_values["every_visit_mc_prediction"],
        [0.9, 0.95, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        result.visit_counts["first_visit_mc_prediction"],
        [1, 1, 0, 0],
    )
    np.testing.assert_array_equal(
        result.visit_counts["every_visit_mc_prediction"],
        [1, 2, 0, 0],
    )


def test_both_methods_approach_exact_values() -> None:
    result = compare_prediction_methods(
        run_id="test-run",
        action_values=ACTION_VALUES,
        episodes=2000,
        epsilon=0.1,
        discount=1.0,
        max_episode_steps=20,
        seed=11,
    )

    for estimates in result.final_values.values():
        np.testing.assert_allclose(estimates[:2], result.true_values[:2], atol=0.04)
    assert (
        result.visit_counts["every_visit_mc_prediction"][1]
        > result.visit_counts["first_visit_mc_prediction"][1]
    )


def test_report_contains_reference_errors_and_visit_comparison() -> None:
    result = compare_prediction_methods(
        run_id="test-run",
        action_values=ACTION_VALUES,
        episodes=10,
        epsilon=0.5,
        discount=1.0,
        max_episode_steps=10,
        seed=3,
    )
    final_estimates = [
        {
            "seed": 3,
            "policy_probabilities": result.policy_probabilities.tolist(),
            "true_values": result.true_values.tolist(),
        }
    ]

    report = render_report(
        run_id="test-run",
        episode_records=result.episode_records,
        trace_records=result.trace_records,
        final_estimates=final_estimates,
        control_episodes=20,
        prediction_episodes=10,
        seeds=[3],
        epsilon=0.5,
        discount=1.0,
        max_episode_steps=10,
    )

    assert "Source run: `test-run`" in report
    assert "Frozen policy and exact values" in report
    assert "Final prediction estimates" in report
    assert "First-visit MC prediction" in report
    assert "Every-visit MC prediction" in report
    assert "Error during learning" in report


@pytest.mark.parametrize(
    "episodes, max_episode_steps, message",
    [
        (0, 10, "episodes must be at least"),
        (1, 0, "max_episode_steps must be at least"),
    ],
)
def test_comparison_rejects_invalid_sizes(
    episodes: int,
    max_episode_steps: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_prediction_methods(
            run_id="test-run",
            action_values=ACTION_VALUES,
            episodes=episodes,
            epsilon=0.1,
            discount=1.0,
            max_episode_steps=max_episode_steps,
            seed=0,
        )
