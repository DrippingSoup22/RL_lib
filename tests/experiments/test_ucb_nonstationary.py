import numpy as np

from experiments.bandits.ucb_nonstationary import run_grid


def test_grid_runs_every_ucb_constant_and_update_rule_condition() -> None:
    results = run_grid(
        k=3,
        exploration_constants=[0.0, 1.0],
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
    assert [result.parameters["c"] for result in results] == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
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
