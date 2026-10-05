import gymnasium as gym
import numpy as np
import pytest
import torch

from experiments.function_approximation.run import (
    algorithm_final_epsilon,
    copy_model_state,
    evaluate_prediction_episode,
    linearly_decayed_epsilon,
    make_agent,
    make_prediction_agent,
    train_episode,
    train_prediction_episode,
)


def test_epsilon_decays_linearly_to_each_algorithms_final_value() -> None:
    assert algorithm_final_epsilon("sarsa", 0.1, 0.01) == 0.01
    assert algorithm_final_epsilon("q_learning", 0.1, 0.01) == 0.1
    values = [
        linearly_decayed_epsilon(0.2, 0.0, episode, training_episodes=5)
        for episode in range(5)
    ]
    assert values == pytest.approx([0.2, 0.15, 0.1, 0.05, 0.0])
    assert linearly_decayed_epsilon(0.2, 0.0, 0, training_episodes=1) == 0.2


@pytest.mark.parametrize("algorithm", ("sarsa", "q_learning"))
def test_function_approximation_runner_flushes_n_step_truncation(
    algorithm: str,
) -> None:
    env = gym.wrappers.RescaleObservation(
        gym.make("MountainCar-v0", max_episode_steps=1),
        np.float32(-1.0),
        np.float32(1.0),
    )
    agent = make_agent(
        algorithm,
        env,
        learning_rate=0.001,
        discount=0.99,
        epsilon=0.1,
        optimizer_name="sgd",
        hidden_sizes=(4,),
        seed=0,
    )

    result = train_episode(
        env,
        agent,
        seed=0,
        rollout_steps=3,
        collect_diagnostics=True,
    )

    assert result.truncated
    assert result.episode_length == 1
    assert result.diagnostics is not None
    assert np.isfinite(result.diagnostics.mean_absolute_td_error)
    env.close()


def test_td_prediction_runner_flushes_n_step_truncation() -> None:
    env = gym.wrappers.RescaleObservation(
        gym.make("MountainCar-v0", max_episode_steps=1),
        np.float32(-1.0),
        np.float32(1.0),
    )
    predictor = make_prediction_agent(
        env,
        learning_rate=0.001,
        discount=0.99,
        optimizer_name="sgd",
        hidden_sizes=(4,),
        seed=0,
    )

    result = train_prediction_episode(
        env,
        predictor,
        environment_seed=0,
        action_seed=0,
        rollout_steps=3,
    )

    assert result.truncated
    assert not result.terminated
    assert result.episode_length == 1
    env.close()


def test_td_prediction_evaluation_is_frozen_and_reports_finite_errors() -> None:
    env = gym.make("CartPole-v1")
    predictor = make_prediction_agent(
        env,
        learning_rate=0.001,
        discount=0.99,
        optimizer_name="sgd",
        hidden_sizes=(4,),
        seed=0,
    )
    before = copy_model_state(predictor.model)

    result = evaluate_prediction_episode(
        env,
        predictor,
        environment_seed=0,
        action_seed=0,
    )

    assert int(result["episode_length"]) > 0
    assert np.all(
        np.isfinite(
            (
                result["mean_absolute_error"],
                result["root_mean_squared_error"],
                result["mean_error"],
            )
        )
    )
    for name, value in predictor.model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    env.close()
