import numpy as np
import pytest

from rl_lib.data import ObservationNormalizer


def test_bounds_normalization_maps_declared_range_to_unit_range() -> None:
    normalizer = ObservationNormalizer(
        2,
        "bounds",
        low=[-2.0, 0.0],
        high=[2.0, 10.0],
    )

    np.testing.assert_allclose(normalizer.normalize([-2.0, 10.0]), [-1.0, 1.0])
    np.testing.assert_allclose(normalizer.normalize([0.0, 5.0]), [0.0, 0.0])


def test_running_normalization_updates_only_when_requested() -> None:
    normalizer = ObservationNormalizer(2, "running")

    np.testing.assert_allclose(normalizer.normalize([2.0, 4.0]), [2.0, 4.0])
    np.testing.assert_allclose(
        normalizer.normalize([2.0, 4.0], update=True),
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        normalizer.normalize([4.0, 8.0], update=True),
        [1.0, 1.0],
        rtol=1e-5,
    )
    count = normalizer.count

    normalizer.normalize([6.0, 12.0])

    assert normalizer.count == count


def test_normalizer_state_round_trip_is_owned_and_reproducible() -> None:
    normalizer = ObservationNormalizer(2, "running")
    normalizer.normalize([1.0, 3.0], update=True)
    normalizer.normalize([5.0, 7.0], update=True)
    state = normalizer.state_dict()
    restored = ObservationNormalizer.from_state_dict(state)

    state["mean"][0] = 100.0

    np.testing.assert_allclose(
        restored.normalize([3.0, 5.0]),
        normalizer.normalize([3.0, 5.0]),
    )
    assert restored.count == 2


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: ObservationNormalizer(0),
        lambda: ObservationNormalizer(2, "unknown"),
        lambda: ObservationNormalizer(2, "bounds"),
        lambda: ObservationNormalizer(2, "bounds", low=[0, 0], high=[1, np.inf]),
        lambda: ObservationNormalizer(2, "bounds", low=[0, 0], high=[1, 0]),
    ),
)
def test_normalizer_rejects_invalid_configuration(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()
