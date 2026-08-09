"""Tests for Monte Carlo experiment measurement and evaluation."""

import numpy as np
import pytest

from experiments.monte_carlo.control import (
    EPISODE_FIELDS,
    evaluate_action_values,
    render_report,
    train_control,
)
from experiments.monte_carlo.looping_mdp import LoopingMDP
from rl_lib.algorithms.tabular.monte_carlo_control import (
    EveryVisitMonteCarloControl,
    FirstVisitMonteCarloControl,
)


@pytest.mark.parametrize(
    "algorithm, control_class",
    [
        ("first_visit_mc_control", FirstVisitMonteCarloControl),
        ("every_visit_mc_control", EveryVisitMonteCarloControl),
    ],
)
def test_train_control_records_episodes_and_learns_preferred_actions(
    algorithm: str,
    control_class: type,
) -> None:
    result = train_control(
        run_id="test-run",
        algorithm=algorithm,
        control_class=control_class,
        episodes=200,
        epsilon=0.1,
        discount=1.0,
        max_episode_steps=20,
        seed=7,
    )

    assert len(result.episode_records) == 200
    assert len(result.training_records) == 200
    assert tuple(result.episode_records[0]) == EPISODE_FIELDS
    assert [record["episode"] for record in result.episode_records] == list(range(200))
    assert all(record["seed"] == 7 for record in result.episode_records)
    assert result.policy_probabilities[LoopingMDP.START, 0] == pytest.approx(0.95)
    assert result.policy_probabilities[LoopingMDP.DECISION, 1] == pytest.approx(0.95)
    assert np.all(result.visit_counts[:2] > 0)


def test_training_records_match_episode_outcomes() -> None:
    result = train_control(
        run_id="test-run",
        algorithm="first_visit_mc_control",
        control_class=FirstVisitMonteCarloControl,
        episodes=20,
        epsilon=0.1,
        discount=1.0,
        max_episode_steps=20,
        seed=3,
    )

    for episode_record, training_record in zip(
        result.episode_records,
        result.training_records,
        strict=True,
    ):
        assert training_record["episode"] == episode_record["episode"]
        assert training_record["success"] == (
            training_record["final_state"] == LoopingMDP.SUCCESS
        )
        assert episode_record["length"] >= 1
        assert episode_record["terminated"] or episode_record["truncated"]


def test_evaluate_action_values_uses_frozen_greedy_policy() -> None:
    action_values = np.array(
        [
            [1.0, -1.0],
            [0.9, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    original_values = action_values.copy()

    records = evaluate_action_values(
        run_id="test-run",
        algorithm="first_visit_mc_control",
        action_values=action_values,
        episodes=10,
        max_episode_steps=20,
        seed=4,
    )

    assert len(records) == 10
    assert all(tuple(record) == EPISODE_FIELDS for record in records)
    assert all(record["phase"] == "evaluation" for record in records)
    assert all(record["return"] == pytest.approx(1.0) for record in records)
    assert all(record["length"] == 2 for record in records)
    assert all(record["success"] for record in records)
    assert all(record["terminated"] and not record["truncated"] for record in records)
    np.testing.assert_array_equal(action_values, original_values)


def test_report_summarizes_performance_and_final_tables() -> None:
    episode_records = []
    training_records = []
    final_estimates = []
    conditions = (
        ("first_visit_mc_control", FirstVisitMonteCarloControl),
        ("every_visit_mc_control", EveryVisitMonteCarloControl),
    )
    for algorithm, control_class in conditions:
        result = train_control(
            run_id="test-run",
            algorithm=algorithm,
            control_class=control_class,
            episodes=20,
            epsilon=0.1,
            discount=1.0,
            max_episode_steps=20,
            seed=5,
        )
        episode_records.extend(result.episode_records)
        episode_records.extend(
            evaluate_action_values(
                run_id="test-run",
                algorithm=algorithm,
                action_values=result.action_values,
                episodes=5,
                max_episode_steps=20,
                seed=5,
            )
        )
        training_records.extend(result.training_records)
        final_estimates.append(
            {
                "algorithm": algorithm,
                "seed": 5,
                "action_values": result.action_values.tolist(),
                "policy_probabilities": result.policy_probabilities.tolist(),
            }
        )

    report = render_report(
        run_id="test-run",
        episode_records=episode_records,
        training_records=training_records,
        final_estimates=final_estimates,
        episodes_per_seed=20,
        evaluation_episodes_per_seed=5,
        seeds=[5],
        epsilon=0.1,
        discount=1.0,
        max_episode_steps=20,
    )

    assert "Source run: `test-run`" in report
    assert "First-visit MC control" in report
    assert "Every-visit MC control" in report
    assert "Late-training performance" in report
    assert "Greedy evaluation performance" in report
    assert "Final learned estimates" in report
    assert "| START |" in report
    assert "| DECISION |" in report


@pytest.mark.parametrize(
    "parameter, value, message",
    [
        ("episodes", 0, "episodes must be at least"),
        ("epsilon", 0.0, "epsilon must be finite"),
        ("epsilon", np.nan, "epsilon must be finite"),
        ("discount", -0.1, "discount must be finite"),
        ("max_episode_steps", 0, "max_episode_steps must be at least"),
    ],
)
def test_train_control_rejects_invalid_parameters(
    parameter: str,
    value: float,
    message: str,
) -> None:
    arguments = {
        "run_id": "test-run",
        "algorithm": "first_visit_mc_control",
        "control_class": FirstVisitMonteCarloControl,
        "episodes": 2,
        "epsilon": 0.1,
        "discount": 1.0,
        "max_episode_steps": 20,
        "seed": 0,
    }
    arguments[parameter] = value

    with pytest.raises(ValueError, match=message):
        train_control(**arguments)
