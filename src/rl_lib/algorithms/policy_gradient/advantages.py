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
    gae_lambda: float | torch.Tensor = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and return targets for ``T`` steps, shape ``(T, W)`` or ``(T,)``.

    Rows are steps in time; columns, if any, are independent environments.
    ``next_values`` is the critic's value of the state after each step: the
    next row's value inside an episode, the final observation's value at a
    time limit, and the next observation's value on the last row. On
    ``terminated`` rows it is ignored, since nothing follows the end of the
    task. ``episode_ended`` marks termination or time limit: the next row
    belongs to a new episode, so its advantage is not carried back.

    Several reward streams that share the same steps, one for each of several
    critics, are estimated at once: ``rewards``, ``values`` and
    ``next_values`` then end with one more dimension, the streams, ``(T, W,
    K)`` or ``(T, K)``, and ``terminated`` and ``episode_ended`` keep the
    steps' shape and apply to every stream.

    Each step's advantage is its TD error plus ``discount × λ`` times the
    next step's advantage. ``gae_lambda`` is one λ for every step, or a
    tensor of the steps' shape with each step's own, how much of the next
    step's advantage it takes: variable λ (Sutton and Barto, Reinforcement
    Learning, 2nd ed., 2018, section 12.8). With λ = 1 on the steps followed
    by more steps of a decision held over several, and the usual λ on the
    step before each new decision, every decision is judged on all the steps
    it held: its TD errors add up to the discounted rewards of its stretch
    plus the value where the next decision starts, as in a semi-Markov
    decision process (Sutton, Precup and Singh, Artificial Intelligence,
    1999), and decisions are mixed with λ as steps are.
    """
    if not 0 <= discount <= 1:
        raise ValueError("Discount must be in [0, 1]")
    if not isinstance(gae_lambda, torch.Tensor) and not 0 <= gae_lambda <= 1:
        raise ValueError("GAE lambda must be in [0, 1]")
    step_shape = terminated.shape
    if (
        len({tensor.shape for tensor in (rewards, values, next_values)}) != 1
        or episode_ended.shape != step_shape
        or rewards.shape[: len(step_shape)] != step_shape
        or rewards.dim() > len(step_shape) + 1
    ):
        raise ValueError("All inputs must have the same shape, but for the streams")
    if isinstance(gae_lambda, torch.Tensor) and gae_lambda.shape != step_shape:
        raise ValueError("A GAE lambda per step needs the steps' shape")

    advantages = torch.zeros_like(rewards)
    next_advantage = 0
    # The episode ends and the lambdas apply alike to every stream.
    stream_shape = step_shape + (1,) * (rewards.dim() - len(step_shape))
    terminated = terminated.float().reshape(stream_shape)
    episode_ended = episode_ended.float().reshape(stream_shape)
    lambdas = (
        torch.as_tensor(gae_lambda, dtype=rewards.dtype, device=rewards.device)
        .expand(step_shape)
        .reshape(stream_shape)
    )

    # Backwards, so that each step's advantage can add the next step's one.
    for row in range(rewards.shape[0] - 1, -1, -1):
        td_error = (
            rewards[row]
            + discount * (1 - terminated[row]) * next_values[row]
            - values[row]
        )
        advantage = (
            td_error
            + discount * lambdas[row] * (1 - episode_ended[row]) * next_advantage
        )

        advantages[row] = advantage
        next_advantage = advantage

    return_targets = advantages + values
    return advantages, return_targets
