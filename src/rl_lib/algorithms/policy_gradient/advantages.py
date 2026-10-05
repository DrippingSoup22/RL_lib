"""Generalized advantage estimation over a block of collected steps."""

import torch


def generalized_advantage_estimates(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    *,
    terminated: torch.Tensor,
    episode_ended: torch.Tensor,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and return targets for ``T`` steps, shape ``(T, W)`` or ``(T,)``.

    Rows are steps in time; columns, if any, are independent environments.
    ``next_values`` is the critic's value of the state after each step: the
    next row's value inside an episode, the final observation's value at a
    time limit, and the next observation's value on the last row. On
    ``terminated`` rows it is ignored, since nothing follows the end of the
    task. ``episode_ended`` marks termination or time limit: the next row
    belongs to a new episode, so its advantage is not carried back.
    """
    if not 0 <= discount <= 1:
        raise ValueError("Discount must be in [0, 1]")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("GAE lambda must be in [0, 1]")
    inputs = (rewards, values, next_values, terminated, episode_ended)
    if len({tensor.shape for tensor in inputs}) != 1:
        raise ValueError("All inputs must have the same shape")

    advantages = torch.zeros_like(rewards)
    next_advantage = 0
    terminated = terminated.float()
    episode_ended = episode_ended.float()

    # Backwards, so that each step's advantage can add the next step's one.
    for row in range(rewards.shape[0] - 1, -1, -1):
        td_error = (
            rewards[row]
            + discount * (1 - terminated[row]) * next_values[row]
            - values[row]
        )
        advantage = (
            td_error + discount * gae_lambda * (1 - episode_ended[row]) * next_advantage
        )

        advantages[row] = advantage
        next_advantage = advantage

    return_targets = advantages + values
    return advantages, return_targets
