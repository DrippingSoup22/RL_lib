import numpy as np
from numpy.typing import ArrayLike, NDArray


def generalized_advantage_estimates(
    rewards: ArrayLike,
    values: ArrayLike,
    final_value: float,
    *,
    terminated: bool,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:

    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)

    if rewards_array.ndim != 1 or values_array.ndim != 1:
        raise ValueError("Rewards and values must be one-dimensional")
    if len(rewards_array) == 0 or len(values_array) == 0:
        raise ValueError("Rewards and values must be non empty!")
    if not len(rewards_array) == len(values_array):
        raise ValueError("Rewards and values must have the same length!")
    if not np.all(np.isfinite(rewards_array)):
        raise ValueError("All rewards must be finite!")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("All values must be finite!")
    if not np.isfinite(final_value):
        raise ValueError("Final value must be finite!")
    if not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("Discount must be finite and in [0, 1]!")
    if not np.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError("Gae lambda must be finite and in [0, 1]!")

    advantages = np.zeros_like(rewards_array, dtype=np.float32)
    next_value = 0 if terminated else final_value
    next_advantage = 0

    for index in range(len(rewards_array) - 1, -1, -1):
        delta = rewards_array[index] + discount * next_value - values_array[index]
        current_advantage = delta + discount * gae_lambda * next_advantage

        advantages[index] = current_advantage
        next_value = values_array[index]
        next_advantage = current_advantage

    return_targets = advantages + values_array

    return advantages, return_targets
