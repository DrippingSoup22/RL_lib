import gymnasium as gym
import numpy as np
import pytest
import torch

from experiments.function_approximation.configuration import parse_config
from experiments.function_approximation.run import (
    EvaluationResult,
    algorithm_final_epsilon,
    copy_model_state,
    evaluation_score,
    linearly_decayed_epsilon,
    make_agent,
    train_episode,
)


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    (("sarsa", 0.01), ("q_learning", 0.1)),
)
def test_control_algorithms_use_their_appropriate_final_epsilon(
    algorithm: str,
    expected: float,
) -> None:
    assert algorithm_final_epsilon(algorithm, 0.1, 0.01) == expected


def test_final_epsilon_rejects_an_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="algorithm"):
        algorithm_final_epsilon("unknown", 0.1, 0.01)


def test_epsilon_decays_linearly_through_training() -> None:
    values = [
        linearly_decayed_epsilon(0.2, 0.0, episode, training_episodes=5)
        for episode in range(5)
    ]

    assert values == pytest.approx([0.2, 0.15, 0.1, 0.05, 0.0])


def test_single_episode_uses_initial_epsilon() -> None:
    assert linearly_decayed_epsilon(0.2, 0.0, 0, training_episodes=1) == 0.2


@pytest.mark.parametrize(
    ("episode_index", "training_episodes"),
    ((0, 0), (-1, 5), (5, 5)),
)
def test_epsilon_schedule_rejects_invalid_episode_positions(
    episode_index: int,
    training_episodes: int,
) -> None:
    with pytest.raises(ValueError):
        linearly_decayed_epsilon(0.2, 0.0, episode_index, training_episodes)


def test_validation_score_prioritizes_success_then_return_then_progress() -> None:
    results = [
        EvaluationResult(-100.0, 100, True, True, False, 1.1),
        EvaluationResult(-500.0, 500, False, False, True, -0.5),
    ]

    assert evaluation_score(results) == pytest.approx((0.5, -300.0, 0.3))

    with pytest.raises(ValueError, match="validation result"):
        evaluation_score([])


def test_model_snapshot_is_independent_and_restorable() -> None:
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(3.0)
    snapshot = copy_model_state(model)

    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    model.load_state_dict(snapshot)

    torch.testing.assert_close(model.weight, torch.tensor([[2.0]]))
    torch.testing.assert_close(model.bias, torch.tensor([3.0]))


def test_observation_scaling_metadata_matches_the_environment_wrapper() -> None:
    mountain_car = parse_config(
        (
            "--preset",
            "quick",
            "--algorithm",
            "q_learning",
            "--environment",
            "MountainCar-v0",
        )
    )
    cart_pole = parse_config(
        (
            "--preset",
            "quick",
            "--algorithm",
            "q_learning",
            "--environment",
            "CartPole-v1",
        )
    )

    assert mountain_car.observations_rescaled
    assert not cart_pole.observations_rescaled


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
