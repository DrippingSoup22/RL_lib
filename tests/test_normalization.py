import pytest
import torch

from rl_lib.normalization import ObservationNormalizer

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def observations(device="cpu") -> torch.Tensor:
    """Ten observations of three values on very different scales."""
    generator = torch.Generator().manual_seed(0)
    scales = torch.tensor([1.0, 10.0, 100.0])
    return (torch.randn(10, 3, generator=generator) * scales + 5.0).to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_running_statistics_are_the_same_whether_a_batch_is_added_whole_or_in_parts(
    device,
) -> None:
    data = observations(device)
    whole = ObservationNormalizer(3, "running", device=device)
    parts = ObservationNormalizer(3, "running", device=device)

    whole.normalize(data, update_statistics=True)
    parts.normalize(data[:4], update_statistics=True)
    parts.normalize(data[4:], update_statistics=True)

    expected_mean = data.double().mean(dim=0)
    expected_squared_deviation_sum = (data.double() - expected_mean).square().sum(0)
    for normalizer in (whole, parts):
        assert normalizer.observation_count == 10
        torch.testing.assert_close(normalizer.mean, expected_mean)
        torch.testing.assert_close(
            normalizer.squared_deviation_sum, expected_squared_deviation_sum
        )


def test_running_normalization_standardizes_and_clips_without_changing_on_request():
    normalizer = ObservationNormalizer(3, "running", clip_limit=2.0)
    data = observations()
    # Nothing to normalise with before any observation was added.
    torch.testing.assert_close(normalizer.normalize(data), data)

    normalizer.normalize(data, update_statistics=True)
    mean = data.double().mean(dim=0)
    standard_deviation = data.double().std(dim=0, unbiased=False)
    far_away = (mean + 100 * standard_deviation).float()
    batch = torch.stack((data[0], far_away))

    result = normalizer.normalize(batch)

    expected_first = ((data[0].double() - mean) / standard_deviation).float()
    torch.testing.assert_close(result[0], expected_first)
    torch.testing.assert_close(result[1], torch.full((3,), 2.0))
    assert result.dtype == torch.float32
    assert normalizer.observation_count == 10  # not updated without the request


def test_bounds_map_low_and_high_to_minus_one_and_one() -> None:
    normalizer = ObservationNormalizer(
        2,
        "bounds",
        low=torch.tensor([-2.0, 0.0], dtype=torch.float64),
        high=torch.tensor([2.0, 10.0]),
    )
    batch = torch.tensor([[[-2.0, 0.0], [2.0, 10.0], [0.0, 5.0]]])

    result = normalizer.normalize(batch)

    torch.testing.assert_close(
        result, torch.tensor([[[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]])
    )


def test_saved_state_is_a_copy_and_restores_the_same_normalizer() -> None:
    normalizer = ObservationNormalizer(3, "running")
    data = observations()
    normalizer.normalize(data[:5], update_statistics=True)

    state = normalizer.state_dict()
    normalizer.normalize(data[5:], update_statistics=True)
    restored = ObservationNormalizer(3, "running")
    restored.load_state_dict(state)
    restored.normalize(data[5:], update_statistics=True)

    assert state["observation_count"] == 5  # unchanged by the later update
    assert restored.observation_count == normalizer.observation_count
    torch.testing.assert_close(restored.normalize(data), normalizer.normalize(data))
    # A new normaliser can also be built from the state alone.
    assert ObservationNormalizer.from_state_dict(state).observation_count == 5
    # Statistics only fit a normaliser of the same size and mode.
    for other in (ObservationNormalizer(4, "running"), ObservationNormalizer(3)):
        with pytest.raises(ValueError):
            other.load_state_dict(state)


def test_constructor_rejects_invalid_settings() -> None:
    bounds = torch.zeros(2), torch.ones(2)
    for arguments, settings in (
        ((0,), {}),
        ((2, "unknown"), {}),
        ((2, "running"), {"epsilon": 0.0}),
        ((2, "running"), {"clip_limit": float("nan")}),
        ((2, "bounds"), {}),
        ((2, "running"), {"low": bounds[0], "high": bounds[1]}),
        ((2, "bounds"), {"low": torch.zeros(3), "high": torch.ones(3)}),
        ((2, "bounds"), {"low": bounds[1], "high": bounds[0]}),
    ):
        with pytest.raises(ValueError):
            ObservationNormalizer(*arguments, **settings)
