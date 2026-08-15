"""One-step tabular temporal-difference control algorithms."""

import numpy as np

from rl_lib.policies.tabular import TabularPolicy, epsilon_soft_probabilities


class SARSA:
    """On-policy one-step SARSA control with an epsilon-soft policy."""

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

        if number_of_states < 1:
            raise ValueError("Number of state must be at least one!")
        if number_of_actions < 1:
            raise ValueError("Number of actions must be at least one!")
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1]!")
        if not 0 <= discount <= 1:
            raise ValueError("Discount must be in [0, 1]!")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in [0, 1]!")
        if not np.isfinite(initial_value):
            raise ValueError("Initial value must be finite!")

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

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int | None,
        terminated: bool,
    ) -> None:

        if not 0 <= state < self.number_of_states:
            raise ValueError("State must stay inside the state table")
        if not 0 <= action < self.number_of_actions:
            raise ValueError("Action must stay inside the action table")
        if not 0 <= next_state < self.number_of_states:
            raise ValueError("Next state must stay inside the state table")
        if not terminated:
            if next_action is None:
                raise ValueError(
                    "next_action is required for a non-terminal transition"
                )
            if not 0 <= next_action < self.number_of_actions:
                raise ValueError("Next action must stay inside the action table!")
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite!")

        # ----- Policy evaluation step -----
        if terminated:
            target = reward
        else:
            target = reward + self.discount * self.Q[next_state, next_action]

        error = target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * error

        # ----- Policy improvement step -----
        probabilities = epsilon_soft_probabilities(self.Q[state], self.epsilon)
        self.policy.set_action_probabilities(state, probabilities)


class QLearning:
    """Off-policy one-step Q-learning with epsilon-soft behavior."""

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

        if number_of_states < 1:
            raise ValueError("Number of state must be at least one!")
        if number_of_actions < 1:
            raise ValueError("Number of actions must be at least one!")
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1]!")
        if not 0 <= discount <= 1:
            raise ValueError("Discount must be in [0, 1]!")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in [0, 1]!")
        if not np.isfinite(initial_value):
            raise ValueError("Initial value must be finite!")

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

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> None:

        if not 0 <= state < self.number_of_states:
            raise ValueError("State must stay inside the state table!")
        if not 0 <= action < self.number_of_actions:
            raise ValueError("Action must stay inside the action table!")
        if not 0 <= next_state < self.number_of_states:
            raise ValueError("Next state must stay inside the state table!")
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite!")

        # ----- Policy evaluation step -----
        if terminated:
            target = reward
        else:
            best_action = np.argmax(self.Q[next_state])
            target = reward + self.discount * self.Q[next_state, best_action]

        error = target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * error

        # Policy improvement step
        probabilities = epsilon_soft_probabilities(self.Q[state], self.epsilon)
        self.policy.set_action_probabilities(state, probabilities)
