import numpy as np

from experiments.bandits.epsilon_nonstationary import run_grid
from experiments.bandits.evaluation import rolling_mean


def test_grid_runs_every_epsilon_and_update_rule_condition() -> None:
    results = run_grid(
        k=3,
        epsilons=[0.0, 0.1],
        initial_value=0.0,
        constant_step_sizes=[0.01, 0.1],
        reward_std=0.0,
        drift_std=0.01,
        steps=5,
        runs=2,
        seed=7,
    )

    assert [result.parameters["update_rule"] for result in results] == [
        "sample_average",
        "constant",
        "constant",
        "sample_average",
        "constant",
        "constant",
    ]
    assert [result.parameters["epsilon"] for result in results] == [
        0.0,
        0.0,
        0.0,
        0.1,
        0.1,
        0.1,
    ]
    assert [result.parameters["step_size"] for result in results] == [
        "1/N",
        0.01,
        0.1,
        "1/N",
        0.01,
        0.1,
    ]
    for result in results:
        measurements = result.measurements
        assert measurements.rewards.shape == (2, 5)
        assert np.all(np.isfinite(measurements.rewards))
        assert np.all(np.isfinite(measurements.estimation_mse))
        assert np.all(measurements.instantaneous_regret >= 0.0)


def test_rolling_mean_smooths_each_run_independently() -> None:
    samples = np.array(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
        ]
    )

    smoothed = rolling_mean(samples, window=2)

    np.testing.assert_array_equal(
        smoothed,
        [
            [1.0, 2.0, 4.0, 6.0],
            [2.0, 3.0, 5.0, 7.0],
        ],
    )
