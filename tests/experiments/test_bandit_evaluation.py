import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.bandits.evaluation import (
    BanditConditionResult,
    evaluate_bandit,
    mean_and_confidence_interval,
    write_convergence_csv,
    write_summary_csv,
)
from experiments.bandits.nonstationary_gaussian_bandit import (
    NonstationaryGaussianBandit,
)
from experiments.bandits.stationary_gaussian_bandit import StationaryGaussianBandit
from rl_lib.algorithms.tabular.epsilon_greedy_bandits import EpsilonGreedyBandits


class DriftingBandit:
    def __init__(self) -> None:
        self.action_values = np.array([1.0, 0.0])

    @property
    def optimal_action(self) -> int:
        return int(np.argmax(self.action_values))

    def step(self, action: int) -> float:
        reward = float(self.action_values[action])
        self.action_values = np.array([0.0, 1.0])
        return reward


class FirstActionAgent:
    def __init__(self) -> None:
        self.estimates = np.zeros(2)
        self.counts = np.zeros(2, dtype=int)

    def select_action(self) -> int:
        return 0

    def update(self, action: int, reward: float) -> None:
        self.estimates[action] = reward
        self.counts[action] += 1


class SecondActionAgent(FirstActionAgent):
    def select_action(self) -> int:
        return 1


def test_evaluation_records_expected_deterministic_measurements() -> None:
    measurements = evaluate_bandit(
        environment_factory=lambda seed: StationaryGaussianBandit(
            k=2,
            reward_std=0.0,
            seed=seed,
        ),
        agent_factory=lambda seed: EpsilonGreedyBandits(
            k=2,
            epsilon=0.0,
            initial_value=10.0,
            seed=seed,
        ),
        runs=2,
        steps=2,
        seed=7,
    )

    assert measurements.rewards.shape == (2, 2)
    np.testing.assert_array_equal(measurements.fraction_actions_tried[:, 0], 0.5)
    np.testing.assert_array_equal(measurements.fraction_actions_tried[:, 1], 1.0)
    assert np.all(np.isfinite(measurements.estimation_mse))


def test_confidence_interval_collapses_for_identical_runs() -> None:
    samples = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])

    mean, lower, upper = mean_and_confidence_interval(samples)

    np.testing.assert_array_equal(mean, [1.0, 2.0])
    np.testing.assert_array_equal(lower, mean)
    np.testing.assert_array_equal(upper, mean)


def test_evaluation_uses_values_from_before_a_nonstationary_step() -> None:
    measurements = evaluate_bandit(
        environment_factory=lambda seed: DriftingBandit(),
        agent_factory=lambda seed: FirstActionAgent(),
        runs=1,
        steps=1,
        seed=0,
    )

    assert measurements.estimation_mse[0, 0] == 0.0
    assert measurements.selected_optimal_action[0, 0]
    assert measurements.instantaneous_regret[0, 0] == 0.0


def test_evaluation_records_instantaneous_pseudo_regret() -> None:
    measurements = evaluate_bandit(
        environment_factory=lambda seed: DriftingBandit(),
        agent_factory=lambda seed: SecondActionAgent(),
        runs=1,
        steps=1,
        seed=0,
    )

    assert measurements.instantaneous_regret[0, 0] == 1.0
    assert not measurements.selected_optimal_action[0, 0]


def test_constant_step_epsilon_greedy_runs_on_nonstationary_bandit() -> None:
    measurements = evaluate_bandit(
        environment_factory=lambda seed: NonstationaryGaussianBandit(
            k=3,
            reward_std=0.0,
            drift_std=0.01,
            seed=seed,
        ),
        agent_factory=lambda seed: EpsilonGreedyBandits(
            k=3,
            epsilon=0.1,
            step_size=0.1,
            seed=seed,
        ),
        runs=2,
        steps=5,
        seed=7,
    )

    assert measurements.rewards.shape == (2, 5)
    assert np.all(np.isfinite(measurements.rewards))
    assert np.all(np.isfinite(measurements.estimation_mse))
    assert np.all(measurements.fraction_actions_tried > 0.0)


def test_confidence_interval_rejects_non_matrix_samples() -> None:
    with pytest.raises(ValueError, match="shape"):
        mean_and_confidence_interval(np.array([1.0, 2.0]))


def test_shared_csv_writers_include_conditions_and_metrics(tmp_path: Path) -> None:
    measurements = evaluate_bandit(
        environment_factory=lambda seed: StationaryGaussianBandit(
            k=2,
            reward_std=0.0,
            seed=seed,
        ),
        agent_factory=lambda seed: EpsilonGreedyBandits(k=2, seed=seed),
        runs=2,
        steps=2,
        seed=3,
    )
    results = [
        BanditConditionResult(
            parameters={"epsilon": 0.1, "initial_value": 0.0},
            measurements=measurements,
        )
    ]

    convergence_path = tmp_path / "convergence.csv"
    summary_path = tmp_path / "summary.csv"
    write_convergence_csv(convergence_path, "epsilon_greedy", results)
    write_summary_csv(summary_path, "epsilon_greedy", results)

    with convergence_path.open(encoding="utf-8", newline="") as file:
        convergence = list(csv.DictReader(file))
    with summary_path.open(encoding="utf-8", newline="") as file:
        summary = list(csv.DictReader(file))

    assert len(convergence) == 2
    assert convergence[0]["algorithm"] == "epsilon_greedy"
    assert convergence[0]["epsilon"] == "0.1"
    assert "std_reward" in convergence[0]
    assert "mean_regret" in convergence[0]
    assert len(summary) == 1
    assert "final_mean_estimation_mse" in summary[0]
    assert "final_mean_regret" in summary[0]
