"""Compatibility with Gymnasium that no experiment-runner test already covers.

The runner tests drive Monte Carlo, temporal-difference, function-approximation,
A2C, A3C, and PPO agents on real environments; these tests cover the project's
own bandit environment, the observation encodings of every family, and
REINFORCE, which no runner test drives.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from experiments.bandits.environments import GaussianBandit
from experiments.function_approximation.environments import (
    make_environment as make_function_approximation_environment,
)
from experiments.monte_carlo.run import inspect_environment
from experiments.policy_gradient.run import (
    generate_episode as generate_policy_gradient_episode,
)
from experiments.policy_gradient.run import make_agent, make_environment
from experiments.temporal_difference.environments import environment_configuration


def test_gaussian_bandit_obeys_the_gymnasium_api_and_drifts() -> None:
    env = GaussianBandit(k=3)
    check_env(env, skip_render_check=True)

    env = GaussianBandit(k=3, reward_std=0.0, drift_std=0.1)
    env.reset(seed=0)
    before = env.action_values.copy()
    env.step(0)
    assert not np.array_equal(env.action_values, before)


def test_toy_text_observations_are_encoded_for_every_family() -> None:
    """Tabular families index states; neural families one-hot encode them."""
    for environment, states, one_hot_size, active_entries in (
        ("Blackjack-v1", 704, 45, 3),
        ("CliffWalking-v1", 48, 48, 1),
        ("FrozenLake-v1", 16, 16, 1),
        ("Taxi-v4", 500, 500, 1),
    ):
        observation, _ = gym.make(environment).reset(seed=0)

        monte_carlo_states, _, monte_carlo_encoder, _ = inspect_environment(environment)
        _, td_states, _, _, td_encoder = environment_configuration(
            environment,
            (0,),
            map_size=4,
            safe_probability=0.8,
            slippery=True,
            max_episode_steps=None,
        )
        assert monte_carlo_states == td_states == states
        assert 0 <= monte_carlo_encoder(observation) < states
        assert 0 <= td_encoder(observation) < states

        for factory in (make_function_approximation_environment, make_environment):
            env = factory(environment)
            one_hot, _ = env.reset(seed=0)
            assert one_hot.shape == (one_hot_size,)
            assert set(np.unique(one_hot)) <= {0, 1}
            assert int(np.sum(one_hot)) == active_entries
            if environment == "CliffWalking-v1":
                assert env.spec.max_episode_steps == 200
            env.close()


@pytest.mark.parametrize("environment", ("CartPole-v1", "MountainCarContinuous-v0"))
def test_reinforce_agents_interact_with_gymnasium(environment: str) -> None:
    for algorithm in ("reinforce", "reinforce_with_baseline"):
        env = make_environment(environment, max_episode_steps=1)
        agent = make_agent(
            algorithm,
            env,
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            optimizer_name="adam",
            weight_decay=0.0,
            discount=0.99,
            entropy_coefficient=0.0,
            hidden_sizes=(4,),
            seed=0,
        )
        episode = generate_policy_gradient_episode(
            env, agent, environment_seed=0, action_seed=0
        )

        loss = agent.update(episode)

        assert len(episode.steps) == 1
        if isinstance(env.action_space, gym.spaces.Box):
            assert episode.steps[0].policy_action is not None
        assert np.isfinite(loss)
        assert all(
            torch.isfinite(parameter).all()
            for parameter in agent.actor_network.parameters()
        )
        env.close()
