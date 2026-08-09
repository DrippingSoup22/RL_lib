"""Tests for Monte Carlo episode generation."""

import gymnasium as gym
import numpy as np

from experiments.monte_carlo.episodes import generate_episode
from experiments.monte_carlo.looping_mdp import LoopingMDP
from rl_lib.algorithms.tabular.policies import TabularPolicy
from rl_lib.data.episode import EpisodeStep, discounted_returns


def test_generate_successful_episode() -> None:
    environment = LoopingMDP()
    policy = TabularPolicy(number_of_states=4, number_of_actions=2, seed=42)
    policy.set_action_probabilities(LoopingMDP.START, [1.0, 0.0])
    policy.set_action_probabilities(LoopingMDP.DECISION, [0.0, 1.0])

    episode = generate_episode(environment, policy, seed=7)

    assert episode.steps == (
        EpisodeStep(LoopingMDP.START, 0, 0.0),
        EpisodeStep(LoopingMDP.DECISION, 1, 1.0),
    )
    assert episode.final_state == LoopingMDP.SUCCESS
    assert episode.terminated
    assert not episode.truncated


def test_generate_failing_episode() -> None:
    environment = LoopingMDP()
    policy = TabularPolicy(number_of_states=4, number_of_actions=2, seed=42)
    policy.set_action_probabilities(LoopingMDP.START, [0.0, 1.0])

    episode = generate_episode(environment, policy)

    assert episode.steps == (EpisodeStep(LoopingMDP.START, 1, -1.0),)
    assert episode.final_state == LoopingMDP.FAILURE
    assert episode.terminated
    assert not episode.truncated


def test_generate_looping_episode_until_time_limit() -> None:
    environment = gym.wrappers.TimeLimit(LoopingMDP(), max_episode_steps=3)
    policy = TabularPolicy(number_of_states=4, number_of_actions=2, seed=42)
    policy.set_action_probabilities(LoopingMDP.START, [1.0, 0.0])
    policy.set_action_probabilities(LoopingMDP.DECISION, [1.0, 0.0])

    episode = generate_episode(environment, policy)

    assert episode.steps == (
        EpisodeStep(LoopingMDP.START, 0, 0.0),
        EpisodeStep(LoopingMDP.DECISION, 0, -0.1),
        EpisodeStep(LoopingMDP.DECISION, 0, -0.1),
    )
    assert episode.final_state == LoopingMDP.DECISION
    assert not episode.terminated
    assert episode.truncated


def test_recorded_rewards_align_with_discounted_returns() -> None:
    environment = LoopingMDP()
    policy = TabularPolicy(number_of_states=4, number_of_actions=2, seed=42)
    policy.set_action_probabilities(LoopingMDP.START, [1.0, 0.0])
    policy.set_action_probabilities(LoopingMDP.DECISION, [0.0, 1.0])
    episode = generate_episode(environment, policy)

    returns = discounted_returns(
        [step.reward for step in episode.steps],
        discount=0.9,
    )

    np.testing.assert_allclose(returns, [0.9, 1.0])


def test_generate_episode_resets_environment_between_calls() -> None:
    environment = LoopingMDP()
    policy = TabularPolicy(number_of_states=4, number_of_actions=2, seed=42)
    policy.set_action_probabilities(LoopingMDP.START, [0.0, 1.0])

    first_episode = generate_episode(environment, policy)
    second_episode = generate_episode(environment, policy)

    assert first_episode == second_episode
