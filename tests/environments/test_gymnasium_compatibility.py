import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from experiments.bandits.environments import GaussianBandit
from experiments.monte_carlo.run import (
    BLACKJACK_ACTIONS,
    BLACKJACK_STATES,
    encode_blackjack_state,
    generate_episode,
    inspect_environment,
)
from rl_lib.algorithms.monte_carlo import FirstVisitMonteCarloControl


def test_gaussian_bandit_obeys_gymnasium_api() -> None:
    env = GaussianBandit(k=3)
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=0)
    transition = env.step(1)
    assert observation == 0
    assert len(transition) == 5


def test_nonstationary_bandit_drifts_after_step() -> None:
    env = GaussianBandit(k=3, reward_std=0.0, drift_std=0.1)
    env.reset(seed=0)
    before = env.action_values.copy()
    env.step(0)
    assert not np.array_equal(env.action_values, before)


def test_monte_carlo_control_interacts_with_blackjack() -> None:
    env = gym.make("Blackjack-v1")
    agent = FirstVisitMonteCarloControl(BLACKJACK_STATES, BLACKJACK_ACTIONS, seed=0)
    episode = generate_episode(
        env,
        agent.select_action,
        seed=0,
        encode_observation=encode_blackjack_state,
    )
    agent.update(episode)
    assert episode.terminated
    assert len(episode.steps) > 0
    assert np.sum(agent.visit_counts) > 0
    env.close()


def test_monte_carlo_control_interacts_with_taxi() -> None:
    states, actions, encoder, _ = inspect_environment("Taxi-v4")
    env = gym.make("Taxi-v4")
    agent = FirstVisitMonteCarloControl(states, actions, seed=0)
    episode = generate_episode(
        env,
        agent.select_action,
        seed=0,
        encode_observation=encoder,
    )
    agent.update(episode)
    assert episode.terminated or episode.truncated
    assert len(episode.steps) > 0
    assert np.sum(agent.visit_counts) > 0
    env.close()


def test_temporal_difference_environments_have_discrete_gymnasium_api() -> None:
    for environment in ("CliffWalking-v1", "FrozenLake-v1"):
        env = gym.make(environment, max_episode_steps=10)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.observation_space.start == 0
        assert env.action_space.start == 0

        observation, _ = env.reset(seed=0)
        transition = env.step(0)
        assert env.observation_space.contains(observation)
        assert len(transition) == 5
        assert isinstance(transition[2], bool)
        assert isinstance(transition[3], bool)
        env.close()
