"""Integration tests for fixed-policy Monte Carlo prediction."""

import gymnasium as gym
import numpy as np

from experiments.monte_carlo.episodes import generate_episode
from experiments.monte_carlo.looping_mdp import LoopingMDP
from rl_lib.algorithms.tabular.monte_carlo_prediction import (
    EveryVisitMonteCarloPrediction,
    FirstVisitMonteCarloPrediction,
)
from rl_lib.algorithms.tabular.policies import policy_from_action_values


def test_prediction_methods_evaluate_same_frozen_epsilon_soft_episode() -> None:
    action_values = np.array(
        [
            [1.0, -1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    policy = policy_from_action_values(action_values, epsilon=0.5, seed=3)
    frozen_probabilities = policy.probabilities.copy()
    environment = gym.wrappers.TimeLimit(LoopingMDP(), max_episode_steps=10)

    try:
        episode = generate_episode(environment, policy, seed=3)
    finally:
        environment.close()

    first_visit = FirstVisitMonteCarloPrediction(
        number_of_states=4,
        discount=1.0,
    )
    every_visit = EveryVisitMonteCarloPrediction(
        number_of_states=4,
        discount=1.0,
    )
    first_visit.update(episode)
    every_visit.update(episode)

    assert [(step.state, step.action) for step in episode.steps] == [
        (LoopingMDP.START, 0),
        (LoopingMDP.DECISION, 0),
        (LoopingMDP.DECISION, 1),
    ]
    np.testing.assert_allclose(
        frozen_probabilities,
        [
            [0.75, 0.25],
            [0.25, 0.75],
            [0.5, 0.5],
            [0.5, 0.5],
        ],
    )
    np.testing.assert_array_equal(policy.probabilities, frozen_probabilities)
    np.testing.assert_allclose(first_visit.values, [0.9, 0.9, 0.0, 0.0])
    np.testing.assert_allclose(every_visit.values, [0.9, 0.95, 0.0, 0.0])
    np.testing.assert_array_equal(first_visit.visit_counts, [1, 1, 0, 0])
    np.testing.assert_array_equal(every_visit.visit_counts, [1, 2, 0, 0])
