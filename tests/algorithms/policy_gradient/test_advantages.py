import pytest
import torch

from rl_lib.algorithms.policy_gradient import generalized_advantage_estimates

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_each_way_an_episode_ends_bootstraps_and_chains_correctly(device) -> None:
    """Three steps (rows) in three environments (columns), all rewards and
    values 1, discount 0.5, lambda 1. Column 0 runs past the window's end;
    column 1 terminates on step 0, where a next value of 100 must be ignored;
    column 2 reaches its time limit on step 0, with a final observation worth 2.
    Steps 1 and 2 of columns 1 and 2 belong to new episodes."""
    ones = torch.ones(3, 3, device=device)
    next_values = torch.tensor(
        [[1.0, 100.0, 2.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=device
    )
    terminated = torch.zeros(3, 3, dtype=torch.bool, device=device)
    terminated[0, 1] = True
    episode_ended = terminated.clone()
    episode_ended[0, 2] = True

    advantages, return_targets = generalized_advantage_estimates(
        ones,
        ones,
        next_values,
        terminated=terminated,
        episode_ended=episode_ended,
        discount=0.5,
        gae_lambda=1.0,
    )

    expected = torch.tensor(
        [[0.875, 0.0, 1.0], [0.75, 0.75, 0.75], [0.5, 0.5, 0.5]], device=device
    )
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(return_targets, expected + 1)


def test_zero_lambda_keeps_only_each_steps_own_td_error() -> None:
    """One trajectory of shape (T,), cut by the window after its last step."""
    no_end = torch.zeros(2, dtype=torch.bool)

    advantages, return_targets = generalized_advantage_estimates(
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.5, 0.25]),
        torch.tensor([0.25, 2.0]),
        terminated=no_end,
        episode_ended=no_end,
        discount=0.9,
        gae_lambda=0.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([0.725, 3.55]))
    torch.testing.assert_close(return_targets, torch.tensor([1.225, 3.8]))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"discount": 1.5}, "Discount"),
        ({"gae_lambda": -0.1}, "lambda"),
        ({"gae_lambda": float("nan")}, "lambda"),
        ({"next_values": torch.zeros(3)}, "same shape"),
    ),
)
def test_invalid_arguments_are_rejected(changes, message) -> None:
    arguments = {
        "rewards": torch.zeros(2),
        "values": torch.zeros(2),
        "next_values": torch.zeros(2),
        "terminated": torch.zeros(2, dtype=torch.bool),
        "episode_ended": torch.zeros(2, dtype=torch.bool),
    } | changes

    with pytest.raises(ValueError, match=message):
        generalized_advantage_estimates(**arguments)
