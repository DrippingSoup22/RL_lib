"""Synchronous advantage actor-critic."""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from rl_lib.data import EpisodeStep, rollout_arrays
from rl_lib.models import (
    DiscretePolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)
from rl_lib.policies import CategoricalPolicy, SquashedGaussianPolicy


class A2C:
    """Synchronous actor-critic for categorical or continuous actions."""

    def __init__(
        self,
        actor_model: DiscretePolicyNetwork | GaussianPolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        critic_model: StateValueNetwork,
        critic_optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
        entropy_coefficient: float = 0.0,
        *,
        action_low: ArrayLike | None = None,
        action_high: ArrayLike | None = None,
    ) -> None:

        if not actor_model.observation_size == critic_model.observation_size:
            raise ValueError("Models must have the same observation size!")
        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and in [0, 1]!")
        if not np.isfinite(entropy_coefficient) or not entropy_coefficient >= 0:
            raise ValueError("Entropy coefficient must be finite and non negative!")

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer
        self.discount = discount
        self.entropy_coefficient = entropy_coefficient
        self.policy: CategoricalPolicy | SquashedGaussianPolicy

        if isinstance(actor_model, DiscretePolicyNetwork):
            if action_low is not None or action_high is not None:
                raise ValueError("Categorical A2C must not receive action bounds")
            self.policy = CategoricalPolicy(actor_model)
        elif isinstance(actor_model, GaussianPolicyNetwork):
            if action_low is None or action_high is None:
                raise ValueError("Continuous A2C requires both action bounds")
            self.policy = SquashedGaussianPolicy(actor_model, action_low, action_high)
        else:
            raise TypeError("Actor model must be discrete or Gaussian")

    def sample_action(
        self,
        observation: ArrayLike,
    ) -> tuple[int, int] | tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return environment and policy actions from one policy sample."""
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
        )

        with torch.no_grad():
            if isinstance(self.policy, CategoricalPolicy):
                action, _ = self.policy.sample(observation_tensor)
                action_index = int(action.item())
                return action_index, action_index

            # Store the latent action for the actor update while interacting
            # with the environment through the bounded transformation.
            environment_action, latent_action, _ = self.policy.sample(
                observation_tensor
            )
            return (
                environment_action.cpu().numpy().astype(np.float32, copy=True),
                latent_action.cpu().numpy().astype(np.float32, copy=True),
            )

    def select_action(self, observation: ArrayLike) -> int | NDArray[np.float32]:
        environment_action, _ = self.sample_action(observation)
        return environment_action

    def _rollout_tensors(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reevaluate the representation sampled by the policy: an action index
        # for categorical control or a latent vector for continuous control.
        if isinstance(self.policy, CategoricalPolicy):
            arrays = rollout_arrays(
                steps,
                final_state,
                observation_size=self.actor_model.observation_size,
                number_of_actions=self.policy.model.number_of_actions,
            )
        else:
            arrays = rollout_arrays(
                steps,
                final_state,
                observation_size=self.actor_model.observation_size,
                action_size=self.policy.model.action_size,
            )
        return (
            torch.as_tensor(arrays.observations),
            torch.as_tensor(arrays.actions),
            torch.as_tensor(arrays.rewards),
            torch.as_tensor(arrays.final_state),
        )

    def update(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
        *,
        terminated: bool,
    ) -> tuple[float, float]:

        (
            observations_tensor,
            policy_actions_tensor,
            rewards_tensor,
            final_state_tensor,
        ) = self._rollout_tensors(steps, final_state)

        returns = torch.empty_like(rewards_tensor)
        with torch.no_grad():
            # True termination has no future value. Any nonterminal rollout
            # bootstraps from the critic at the state following its last step.
            if terminated:
                running_return = rewards_tensor.new_zeros(())
            else:
                running_return = self.critic_model(final_state_tensor)

            for index in range(len(rewards_tensor) - 1, -1, -1):
                running_return = rewards_tensor[index] + self.discount * running_return
                returns[index] = running_return

        log_probabilities, entropy = self.policy.evaluate_actions(
            observations_tensor,
            policy_actions_tensor,
        )

        values = self.critic_model(observations_tensor)
        advantages = returns - values

        # The actor treats the advantage as a fixed learning signal; the critic
        # learns the same error through its separate squared-loss update.
        actor_loss = -(log_probabilities * advantages.detach()).mean()
        actor_loss -= self.entropy_coefficient * entropy.mean()

        critic_loss = 0.5 * advantages.square().mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return float(actor_loss.item()), float(critic_loss.item())
