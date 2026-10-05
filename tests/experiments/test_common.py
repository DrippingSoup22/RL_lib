import gymnasium as gym

from experiments.common import finite_state_encoding


def test_finite_state_encoding_indexes_discrete_spaces_and_tuples() -> None:
    states, encode = finite_state_encoding(gym.spaces.Discrete(3, start=4))
    assert states == 3
    assert [encode(value) for value in (4, 5, 6)] == [0, 1, 2]

    states, encode = finite_state_encoding(
        gym.spaces.Tuple(
            (gym.spaces.Discrete(32), gym.spaces.Discrete(11), gym.spaces.Discrete(2))
        )
    )
    assert states == 704
    assert encode((20, 7, 1)) == (20 * 11 + 7) * 2 + 1
