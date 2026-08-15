"""On-policy Monte Carlo control for discrete state and action spaces."""

import numpy as np

from rl_lib.data.episode import Episode, discounted_returns
from rl_lib.policies.tabular import TabularPolicy, epsilon_soft_probabilities


class FirstVisitMonteCarloControl:
    """First-visit Monte Carlo control with an epsilon-soft policy."""

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        epsilon: float = 0.1,
        discount: float = 1.0,
        seed: int | None = None,
    ) -> None:

        if number_of_states < 1:
            raise ValueError("number_of_states must be at least 1")
        if number_of_actions < 1:
            raise ValueError("number_of_actions must be at least 1")
        if not 0 < epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")
        if not 0 <= discount <= 1:
            raise ValueError("Discount must be between 0 and 1")

        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.discount = discount

        self.policy = TabularPolicy(
            number_of_states,
            number_of_actions,
            seed,
        )

        self.Q = np.zeros(
            (number_of_states, number_of_actions),
            dtype=float,
        )
        self.visit_counts = np.zeros(
            (number_of_states, number_of_actions),
            dtype=int,
        )

    def select_action(self, state: int) -> int:
        return self.policy.select_action(state)

    def update(self, episode: Episode) -> None:
        """Update action values and improve the policy from one episode."""

        # Monte Carlo policy evaluation
        rewards = [step.reward for step in episode.steps]
        episode_returns = discounted_returns(rewards, self.discount)
        first_visited_pairs = set()
        first_visited_states = set()

        for step, episode_return in zip(episode.steps, episode_returns, strict=True):
            state = step.state
            action = step.action
            if not 0 <= state < self.number_of_states:
                raise ValueError("Invalid state!")
            if not 0 <= action < self.number_of_actions:
                raise ValueError("Invalid action!")
            pair = (state, action)

            if pair not in first_visited_pairs:
                first_visited_pairs.add(pair)
                first_visited_states.add(state)
                self.visit_counts[state, action] += 1
                self.Q[state, action] += (
                    episode_return - self.Q[state, action]
                ) / self.visit_counts[state, action]

        # Monte Carlo policy improvement
        for state in first_visited_states:
            probabilities = epsilon_soft_probabilities(
                self.Q[state],
                self.epsilon,
            )

            self.policy.set_action_probabilities(
                state,
                probabilities,
            )


class EveryVisitMonteCarloControl:
    """Every-visit Monte Carlo control with an epsilon-soft policy."""

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        epsilon: float = 0.1,
        discount: float = 1.0,
        seed: int | None = None,
    ) -> None:

        if number_of_states < 1:
            raise ValueError("number_of_states must be at least 1")
        if number_of_actions < 1:
            raise ValueError("number_of_actions must be at least 1")
        if not 0 < epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")
        if not 0 <= discount <= 1:
            raise ValueError("Discount must be between 0 and 1")

        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.discount = discount

        self.policy = TabularPolicy(
            number_of_states,
            number_of_actions,
            seed,
        )

        self.Q = np.zeros(
            (number_of_states, number_of_actions),
            dtype=float,
        )
        self.visit_counts = np.zeros(
            (number_of_states, number_of_actions),
            dtype=int,
        )

    def select_action(self, state: int) -> int:
        return self.policy.select_action(state)

    def update(self, episode: Episode) -> None:
        """Update action values and improve the policy from one episode."""

        # Monte Carlo policy evaluation
        rewards = [step.reward for step in episode.steps]
        episode_returns = discounted_returns(rewards, self.discount)
        visited_states = set()

        for step, episode_return in zip(episode.steps, episode_returns, strict=True):
            state = step.state
            action = step.action
            if not 0 <= state < self.number_of_states:
                raise ValueError("Invalid state!")
            if not 0 <= action < self.number_of_actions:
                raise ValueError("Invalid action!")

            visited_states.add(state)
            self.visit_counts[state, action] += 1
            self.Q[state, action] += (
                episode_return - self.Q[state, action]
            ) / self.visit_counts[state, action]

        # Monte Carlo policy improvement
        for state in visited_states:
            probabilities = epsilon_soft_probabilities(
                self.Q[state],
                self.epsilon,
            )

            self.policy.set_action_probabilities(
                state,
                probabilities,
            )
