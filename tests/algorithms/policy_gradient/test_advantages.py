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


def test_several_reward_streams_are_estimated_each_as_alone() -> None:
    """Two streams over the same three steps in two environments, the second
    environment's episode ending on step 0: each stream's estimates are those
    it would get alone."""
    generator = torch.Generator().manual_seed(0)
    rewards, values, next_values = torch.randn(3, 3, 2, 2, generator=generator)
    ended = torch.tensor([[False, True], [False, False], [False, False]])
    settings = {"terminated": ended, "episode_ended": ended, "discount": 0.9}

    advantages, return_targets = generalized_advantage_estimates(
        rewards, values, next_values, **settings
    )

    for stream in range(2):
        alone = generalized_advantage_estimates(
            rewards[..., stream],
            values[..., stream],
            next_values[..., stream],
            **settings,
        )
        torch.testing.assert_close(advantages[..., stream], alone[0])
        torch.testing.assert_close(return_targets[..., stream], alone[1])


def test_a_lambda_per_step_judges_each_held_decision_on_its_whole_stretch() -> None:
    """One trajectory of five steps: a decision on step 0 holds three steps,
    the next, on step 3, two, until the window's end. With lambda 1 inside
    each stretch and 0.5 before the next decision, each decision's advantage
    is its stretch's discounted rewards plus the value where the next one
    starts, minus its own value, plus half the next decision's advantage."""
    generator = torch.Generator().manual_seed(0)
    rewards, values = torch.randn(2, 5, generator=generator)
    last_next_value = torch.randn(1, generator=generator)
    next_values = torch.cat((values[1:], last_next_value))
    no_end = torch.zeros(5, dtype=torch.bool)
    gamma = 0.9

    advantages, _ = generalized_advantage_estimates(
        rewards,
        values,
        next_values,
        terminated=no_end,
        episode_ended=no_end,
        discount=gamma,
        gae_lambda=torch.tensor([1.0, 1.0, 0.5, 1.0, 0.5]),
    )

    second = rewards[3] + gamma * rewards[4] + gamma**2 * last_next_value - values[3]
    first = (
        rewards[0]
        + gamma * rewards[1]
        + gamma**2 * rewards[2]
        + gamma**3 * values[3]
        - values[0]
        + gamma**3 * 0.5 * second
    )
    torch.testing.assert_close(advantages[[0, 3]], torch.cat((first, second)))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"discount": 1.5}, "Discount"),
        ({"gae_lambda": -0.1}, "lambda"),
        ({"gae_lambda": float("nan")}, "lambda"),
        ({"gae_lambda": torch.ones(3)}, "steps' shape"),
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
