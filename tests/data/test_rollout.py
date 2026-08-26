import numpy as np
import pytest

from rl_lib.data import EpisodeStep, rollout_arrays


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


@pytest.mark.parametrize(
    ("steps", "message"),
    (
        ((), "at least one step"),
        ((EpisodeStep(np.array([1.0]), 0.5, 1.0),), "integers"),
        ((EpisodeStep(np.array([1.0]), 2, 1.0),), "action space"),
    ),
)
def test_rollout_arrays_rejects_invalid_steps(steps: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rollout_arrays(
            steps,
            [0.0],
            observation_size=1,
            number_of_actions=2,
        )
