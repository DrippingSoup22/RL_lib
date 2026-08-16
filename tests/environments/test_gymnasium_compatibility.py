import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from experiments.bandits.environments import GaussianBandit
from experiments.monte_carlo.run import (
    BLACKJACK_ACTIONS,
    BLACKJACK_STATES,
    encode_blackjack_state,
    inspect_environment,
)
from experiments.monte_carlo.run import (
    generate_episode as generate_monte_carlo_episode,
)
from experiments.policy_gradient.run import (
    generate_episode as generate_policy_gradient_episode,
)
from experiments.policy_gradient.run import (
    make_agent,
    make_environment,
)
from rl_lib.algorithms.function_approximation import (
    SemiGradientQLearning,
    SemiGradientSARSA,
)
from rl_lib.algorithms.monte_carlo import FirstVisitMonteCarloControl
from rl_lib.algorithms.policy_gradient import A2C, ReinforceWithBaseline
from rl_lib.data import EpisodeStep
from rl_lib.models import ActionValueNetwork


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
    episode = generate_monte_carlo_episode(
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
    episode = generate_monte_carlo_episode(
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


@pytest.mark.parametrize("environment", ("MountainCar-v0", "Acrobot-v1"))
def test_function_approximation_environments_have_compatible_spaces(
    environment: str,
) -> None:
    env = gym.make(environment)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape in ((2,), (6,))
    assert np.all(np.isfinite(env.observation_space.low))
    assert np.all(np.isfinite(env.observation_space.high))
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.start == 0
    assert env.action_space.n == 3

    observation, _ = env.reset(seed=0)
    next_observation, _, terminated, truncated, _ = env.step(0)
    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


@pytest.mark.parametrize("agent_class", (SemiGradientSARSA, SemiGradientQLearning))
def test_function_approximation_agents_interact_with_gymnasium(
    agent_class: type[SemiGradientSARSA] | type[SemiGradientQLearning],
) -> None:
    env = gym.wrappers.RescaleObservation(
        gym.make("MountainCar-v0", max_episode_steps=1),
        np.float32(-1.0),
        np.float32(1.0),
    )
    model = ActionValueNetwork(2, 3, hidden_sizes=())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    agent = agent_class(model, optimizer, seed=0)
    observation, _ = env.reset(seed=0)
    action = agent.select_action(observation)
    next_observation, reward, terminated, truncated, _ = env.step(action)

    if isinstance(agent, SemiGradientSARSA):
        next_action = None if terminated else agent.select_action(next_observation)
        agent.update(
            (EpisodeStep(observation, action, reward),),
            next_observation,
            next_action,
            terminated=terminated,
        )
    else:
        agent.update(
            (EpisodeStep(observation, action, reward),),
            next_observation,
            terminated=terminated,
        )

    assert truncated
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    env.close()


@pytest.mark.parametrize(
    "algorithm",
    ("reinforce", "reinforce_with_baseline", "a2c"),
)
def test_policy_gradient_agents_interact_with_gymnasium(algorithm: str) -> None:
    env = make_environment("CartPole-v1", max_episode_steps=1)
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
        env,
        agent,
        environment_seed=0,
        action_seed=0,
    )

    if isinstance(agent, A2C):
        losses = agent.update(
            episode.steps,
            episode.final_state,
            terminated=episode.terminated,
        )
    else:
        losses = (agent.update(episode),)

    assert episode.terminated or episode.truncated
    assert len(episode.steps) == 1
    assert np.all(np.isfinite(losses))
    assert all(
        torch.isfinite(parameter).all() for parameter in agent.actor_model.parameters()
    )
    if isinstance(agent, (ReinforceWithBaseline, A2C)):
        assert all(
            torch.isfinite(parameter).all()
            for parameter in agent.critic_model.parameters()
        )
    env.close()
