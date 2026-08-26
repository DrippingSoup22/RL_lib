"""Semi-gradient control with bootstrapped rollout returns."""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from rl_lib.data import EpisodeStep, rollout_arrays
from rl_lib.models import ActionValueNetwork
from rl_lib.policies import epsilon_soft_probabilities


class _SemiGradientControl:
    def __init__(
        self,
        model: ActionValueNetwork,
        optimizer: torch.optim.Optimizer,
        discount: float,
        epsilon: float,
        seed: int | None,
    ) -> None:
        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("Discount must be finite and in [0, 1]")
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be finite and in [0, 1]")
        self.model = model
        self.optimizer = optimizer
        self.discount = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: ArrayLike) -> int:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        if observation_tensor.shape != (self.model.observation_size,):
            raise ValueError("Observation must match the model input size")
        with torch.no_grad():
            action_values = self.model(observation_tensor)
        probabilities = epsilon_soft_probabilities(action_values.numpy(), self.epsilon)
        return int(self.rng.choice(self.model.number_of_actions, p=probabilities))

    def _rollout_tensors(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        arrays = rollout_arrays(
            steps,
            final_state,
            observation_size=self.model.observation_size,
            number_of_actions=self.model.number_of_actions,
        )
        return (
            torch.as_tensor(arrays.observations),
            torch.as_tensor(arrays.actions),
            torch.as_tensor(arrays.rewards),
            torch.as_tensor(arrays.final_state),
        )

    def _update(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
    ) -> tuple[float, ...]:
        returns = torch.empty_like(rewards)
        running_return = bootstrap
        with torch.no_grad():
            for index in range(len(steps) - 1, -1, -1):
                running_return = rewards[index] + self.discount * running_return
                returns[index] = running_return

        action_values = self.model(observations)
        selected_values = action_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        errors = returns - selected_values
        loss = 0.5 * errors.square().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return tuple(float(error) for error in errors.detach().tolist())


class SemiGradientSARSA(_SemiGradientControl):
    """On-policy semi-gradient SARSA over bounded rollouts."""

    def __init__(
        self,
        model: ActionValueNetwork,
        optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(model, optimizer, discount, epsilon, seed)

    def update(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
        final_action: int | None,
        *,
        terminated: bool,
    ) -> tuple[float, ...]:
        observations, actions, rewards, final_state_tensor = self._rollout_tensors(
            steps, final_state
        )
        with torch.no_grad():
            if terminated:
                bootstrap = rewards.new_zeros(())
            else:
                if final_action is None:
                    raise ValueError("Final action is required to bootstrap SARSA")
                if not 0 <= final_action < self.model.number_of_actions:
                    raise ValueError("Final action must stay inside the action space")
                bootstrap = self.model(final_state_tensor)[final_action]
        return self._update(steps, observations, actions, rewards, bootstrap)


class SemiGradientQLearning(_SemiGradientControl):
    """Off-policy semi-gradient Q-learning over bounded rollouts."""

    def __init__(
        self,
        model: ActionValueNetwork,
        optimizer: torch.optim.Optimizer,
        discount: float = 1.0,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(model, optimizer, discount, epsilon, seed)

    def update(
        self,
        steps: Sequence[EpisodeStep[NDArray[np.float32]]],
        final_state: ArrayLike,
        *,
        terminated: bool,
    ) -> tuple[float, ...]:
        observations, actions, rewards, final_state_tensor = self._rollout_tensors(
            steps, final_state
        )
        with torch.no_grad():
            bootstrap = (
                rewards.new_zeros(())
                if terminated
                else torch.max(self.model(final_state_tensor))
            )
        return self._update(steps, observations, actions, rewards, bootstrap)
