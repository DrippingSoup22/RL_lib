"""Asynchronous advantage actor-critic."""

from collections.abc import Sequence
from contextlib import AbstractContextManager

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike, NDArray

from rl_lib.data import EpisodeStep, rollout_arrays
from rl_lib.models import (
    DiscretePolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)
from rl_lib.policies import CategoricalPolicy, SquashedGaussianPolicy


def _validate_model_pair(
    local_model: nn.Module,
    shared_model: nn.Module,
    name: str,
) -> None:
    local_parameters = tuple(local_model.parameters())
    shared_parameters = tuple(shared_model.parameters())

    if tuple(parameter.shape for parameter in local_parameters) != tuple(
        parameter.shape for parameter in shared_parameters
    ):
        raise ValueError(f"Local and shared {name} must have matching parameters")

    if any(
        local.data_ptr() == shared.data_ptr()
        for local, shared in zip(
            local_parameters,
            shared_parameters,
            strict=True,
        )
    ):
        raise ValueError(f"Local and shared {name} must use separate parameters")


def _copy_gradients(local_model: nn.Module, shared_model: nn.Module) -> None:
    for local_parameter, shared_parameter in zip(
        local_model.parameters(),
        shared_model.parameters(),
        strict=True,
    ):
        if local_parameter.grad is None:
            raise RuntimeError("Local parameter has no gradient!")

        shared_parameter.grad = local_parameter.grad.detach().clone()


class A3C:
    """Local A3C worker backed by shared categorical or continuous models."""

    def __init__(
        self,
        actor_model: DiscretePolicyNetwork | GaussianPolicyNetwork,
        shared_actor_model: DiscretePolicyNetwork | GaussianPolicyNetwork,
        shared_actor_optimizer: torch.optim.Optimizer,
        critic_model: StateValueNetwork,
        shared_critic_model: StateValueNetwork,
        shared_critic_optimizer: torch.optim.Optimizer,
        update_lock: AbstractContextManager[object],
        discount: float = 1.0,
        entropy_coefficient: float = 0.0,
        *,
        action_low: ArrayLike | None = None,
        action_high: ArrayLike | None = None,
    ) -> None:

        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and in [0, 1]!")
        if not np.isfinite(entropy_coefficient) or entropy_coefficient < 0:
            raise ValueError("Entropy coefficient must be finite and non negative!")

        observation_sizes = {
            actor_model.observation_size,
            critic_model.observation_size,
            shared_actor_model.observation_size,
            shared_critic_model.observation_size,
        }
        if len(observation_sizes) != 1:
            raise ValueError("All models must have the same observation size")

        if isinstance(actor_model, DiscretePolicyNetwork):
            if not isinstance(shared_actor_model, DiscretePolicyNetwork):
                raise ValueError(
                    "Local and shared actors must use the same policy type"
                )
            if actor_model.number_of_actions != shared_actor_model.number_of_actions:
                raise ValueError(
                    "Local and shared actors must have the same number of actions"
                )
            if action_low is not None or action_high is not None:
                raise ValueError("Categorical A3C must not receive action bounds")
            policy: CategoricalPolicy | SquashedGaussianPolicy = CategoricalPolicy(
                actor_model
            )
        elif isinstance(actor_model, GaussianPolicyNetwork):
            if not isinstance(shared_actor_model, GaussianPolicyNetwork):
                raise ValueError(
                    "Local and shared actors must use the same policy type"
                )
            if actor_model.action_size != shared_actor_model.action_size:
                raise ValueError(
                    "Local and shared actors must have the same action size"
                )
            if action_low is None or action_high is None:
                raise ValueError("Continuous A3C requires both action bounds")
            policy = SquashedGaussianPolicy(actor_model, action_low, action_high)
        else:
            raise TypeError("Actor models must be discrete or Gaussian")

        _validate_model_pair(actor_model, shared_actor_model, "actors")
        _validate_model_pair(critic_model, shared_critic_model, "critics")

        self.actor_model = actor_model
        self.shared_actor_model = shared_actor_model
        self.shared_actor_optimizer = shared_actor_optimizer
        self.critic_model = critic_model
        self.shared_critic_model = shared_critic_model
        self.shared_critic_optimizer = shared_critic_optimizer
        self.update_lock = update_lock
        self.discount = discount
        self.entropy_coefficient = entropy_coefficient
        self.policy = policy

        with self.update_lock:
            self._synchronize()

    def _synchronize(self) -> None:
        self.actor_model.load_state_dict(self.shared_actor_model.state_dict())
        self.critic_model.load_state_dict(self.shared_critic_model.state_dict())

    def sample_action(
        self,
        observation: ArrayLike,
    ) -> tuple[int, int] | tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return environment and policy actions from one local-policy sample."""
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
        )

        with torch.no_grad():
            if isinstance(self.policy, CategoricalPolicy):
                action, _ = self.policy.sample(observation_tensor)
                action_index = int(action.item())
                return action_index, action_index

            # The worker sends bounded actions to Gymnasium and retains latent
            # actions for the local policy-gradient calculation.
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
        # Preserve categorical indices or Gaussian latent vectors so the local
        # actor differentiates the same actions that generated the rollout.
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

    def _rollout_losses(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
        *,
        terminated: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        (
            observations_tensor,
            policy_actions_tensor,
            rewards_tensor,
            final_state_tensor,
        ) = self._rollout_tensors(steps, final_state)

        with torch.no_grad():
            # True termination has no future value. Any nonterminal rollout
            # bootstraps from the local critic beyond its final collected step.
            if terminated:
                running_return = rewards_tensor.new_zeros(())
            else:
                running_return = self.critic_model(final_state_tensor)

            returns = torch.empty_like(rewards_tensor)
            for index in range(len(rewards_tensor) - 1, -1, -1):
                running_return = rewards_tensor[index] + self.discount * running_return
                returns[index] = running_return

        log_probabilities, entropy = self.policy.evaluate_actions(
            observations_tensor,
            policy_actions_tensor,
        )

        values = self.critic_model(observations_tensor)
        advantages = returns - values

        # Only the local actor and critic build gradients here. Their gradients
        # are copied to the shared models during the locked optimizer step.
        actor_loss = -(log_probabilities * advantages.detach()).mean()
        actor_loss -= self.entropy_coefficient * entropy.mean()

        critic_loss = 0.5 * advantages.square().mean()

        return actor_loss, critic_loss

    def update(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
        *,
        terminated: bool,
    ) -> tuple[float, float]:

        actor_loss, critic_loss = self._rollout_losses(
            steps, final_state, terminated=terminated
        )

        self.actor_model.zero_grad(set_to_none=True)
        self.critic_model.zero_grad(set_to_none=True)
        actor_loss.backward()
        critic_loss.backward()

        with self.update_lock:
            self.shared_actor_optimizer.zero_grad(set_to_none=True)
            self.shared_critic_optimizer.zero_grad(set_to_none=True)
            _copy_gradients(self.actor_model, self.shared_actor_model)
            _copy_gradients(self.critic_model, self.shared_critic_model)
            self.shared_actor_optimizer.step()
            self.shared_critic_optimizer.step()
            self._synchronize()

        return float(actor_loss.item()), float(critic_loss.item())
