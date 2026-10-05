import numpy as np

from rl_lib.trajectories import EpisodeStep, rollout_arrays


def test_rollout_arrays_converts_episode_steps_to_typed_arrays() -> None:
    arrays = rollout_arrays(
        (
            EpisodeStep(np.array([1.0, 2.0]), 0, 3.0),
            EpisodeStep(np.array([4.0, 5.0]), 1, 6.0),
        ),
        [7.0, 8.0],
        observation_size=2,
        number_of_actions=2,
    )

    np.testing.assert_array_equal(arrays.observations, [[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_array_equal(arrays.actions, [0, 1])
    np.testing.assert_array_equal(arrays.rewards, [3.0, 6.0])
    np.testing.assert_array_equal(arrays.final_state, [7.0, 8.0])
    assert arrays.observations.dtype == np.float32
    assert arrays.actions.dtype == np.int64
    assert arrays.rewards.dtype == np.float32


def test_rollout_arrays_retains_continuous_latent_actions() -> None:
    arrays = rollout_arrays(
        (
            EpisodeStep(
                np.array([1.0, 2.0]),
                np.array([0.5], dtype=np.float32),
                3.0,
                policy_action=np.array([0.25], dtype=np.float32),
            ),
            EpisodeStep(
                np.array([4.0, 5.0]),
                np.array([-0.5], dtype=np.float32),
                6.0,
                policy_action=np.array([-0.25], dtype=np.float32),
            ),
        ),
        [7.0, 8.0],
        observation_size=2,
        action_size=1,
    )

    np.testing.assert_array_equal(arrays.actions, [[0.25], [-0.25]])
    assert arrays.actions.dtype == np.float32
