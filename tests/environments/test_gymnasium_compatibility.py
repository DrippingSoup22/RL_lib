import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from experiments.bandits.environments import GaussianBandit
from experiments.monte_carlo.run import (
    NUMBER_OF_ACTIONS,
    NUMBER_OF_STATES,
    generate_episode,
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
    agent = FirstVisitMonteCarloControl(NUMBER_OF_STATES, NUMBER_OF_ACTIONS, seed=0)
    episode = generate_episode(env, agent.select_action, seed=0)
    agent.update(episode)
    assert episode.terminated
    assert len(episode.steps) > 0
    assert np.sum(agent.visit_counts) > 0
    env.close()
