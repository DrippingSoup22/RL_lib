"""Tabular temporal-difference control with rollout returns."""

from collections.abc import Sequence

import numpy as np

from rl_lib.data import EpisodeStep
from rl_lib.policies.tabular import TabularPolicy, epsilon_soft_probabilities


class _TabularControl:
    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        learning_rate: float,
        discount: float,
        epsilon: float,
        initial_value: float,
        seed: int | None,
    ) -> None:
        if number_of_states < 1:
            raise ValueError("Number of states must be at least one")
        if number_of_actions < 1:
            raise ValueError("Number of actions must be at least one")
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1]")
        if not 0 <= discount <= 1:
            raise ValueError("Discount must be in [0, 1]")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in [0, 1]")
        if not np.isfinite(initial_value):
            raise ValueError("Initial value must be finite")

        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.Q = np.full(
            (number_of_states, number_of_actions), initial_value, dtype=float
        )
        self.policy = TabularPolicy(number_of_states, number_of_actions, seed)

    def select_action(self, state: int) -> int:
        return self.policy.select_action(state)

    def _validate_rollout(
        self,
        steps: Sequence[EpisodeStep[int]],
        final_state: int,
    ) -> None:
        if not steps:
            raise ValueError("Steps must not be empty")
        if not 0 <= final_state < self.number_of_states:
            raise ValueError("Final state must stay inside the state table")
        for step in steps:
            if not 0 <= step.state < self.number_of_states:
                raise ValueError("State must stay inside the state table")
            if not 0 <= step.action < self.number_of_actions:
                raise ValueError("Action must stay inside the action table")
            if not np.isfinite(step.reward):
                raise ValueError("Reward must be finite")

    def _update_from_bootstrap(
        self,
        steps: Sequence[EpisodeStep[int]],
        bootstrap: float,
    ) -> tuple[float, ...]:
        returns = np.empty(len(steps), dtype=float)
        running_return = bootstrap
        for index in range(len(steps) - 1, -1, -1):
            running_return = steps[index].reward + self.discount * running_return
            returns[index] = running_return

        errors = []
        for step, target in zip(steps, returns, strict=True):
            error = target - self.Q[step.state, step.action]
            self.Q[step.state, step.action] += self.learning_rate * error
            probabilities = epsilon_soft_probabilities(self.Q[step.state], self.epsilon)
            self.policy.set_action_probabilities(step.state, probabilities)
            errors.append(float(error))
        return tuple(errors)


class SARSA(_TabularControl):
    """On-policy SARSA using bootstrapped rollout returns."""

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        learning_rate: float = 0.1,
        discount: float = 1.0,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            number_of_states,
            number_of_actions,
            learning_rate,
            discount,
            epsilon,
            initial_value,
            seed,
        )

    def update(
        self,
        steps: Sequence[EpisodeStep[int]],
        final_state: int,
        final_action: int | None,
        *,
        terminated: bool,
    ) -> tuple[float, ...]:
        self._validate_rollout(steps, final_state)
        if terminated:
            bootstrap = 0.0
        else:
            if final_action is None:
                raise ValueError("Final action is required to bootstrap SARSA")
            if not 0 <= final_action < self.number_of_actions:
                raise ValueError("Final action must stay inside the action table")
            bootstrap = float(self.Q[final_state, final_action])
        return self._update_from_bootstrap(steps, bootstrap)


class QLearning(_TabularControl):
    """Off-policy Q-learning using bootstrapped rollout returns."""

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        learning_rate: float = 0.1,
        discount: float = 1.0,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            number_of_states,
            number_of_actions,
            learning_rate,
            discount,
            epsilon,
            initial_value,
            seed,
        )

    def update(
        self,
        steps: Sequence[EpisodeStep[int]],
        final_state: int,
        *,
        terminated: bool,
    ) -> tuple[float, ...]:
        self._validate_rollout(steps, final_state)
        bootstrap = 0.0 if terminated else float(np.max(self.Q[final_state]))
        return self._update_from_bootstrap(steps, bootstrap)
