import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

from experiments.monte_carlo.looping_mdp import LoopingMDP


def test_environment_passes_gymnasium_checker() -> None:
    check_env(LoopingMDP(), skip_render_check=True)


def test_reset_starts_a_new_episode_at_normal_start() -> None:
    environment = LoopingMDP()

    state, info = environment.reset(seed=42)

    assert state == LoopingMDP.START
    assert info == {}


def test_start_action_zero_reaches_decision_state() -> None:
    environment = LoopingMDP()
    environment.reset()

    transition = environment.step(0)

    assert transition == (LoopingMDP.DECISION, 0.0, False, False, {})


def test_start_action_one_reaches_failure() -> None:
    environment = LoopingMDP()
    environment.reset()

    transition = environment.step(1)

    assert transition == (LoopingMDP.FAILURE, -1.0, True, False, {})


def test_decision_action_zero_loops_with_step_cost() -> None:
    environment = LoopingMDP()
    environment.reset()
    environment.step(0)

    transition = environment.step(0)

    assert transition == (LoopingMDP.DECISION, -0.1, False, False, {})


def test_decision_action_one_reaches_success() -> None:
    environment = LoopingMDP()
    environment.reset()
    environment.step(0)

    transition = environment.step(1)

    assert transition == (LoopingMDP.SUCCESS, 1.0, True, False, {})


def test_step_requires_reset() -> None:
    environment = LoopingMDP()

    with pytest.raises(RuntimeError, match="reset must be called before"):
        environment.step(0)


def test_step_after_termination_requires_reset() -> None:
    environment = LoopingMDP()
    environment.reset()
    environment.step(1)

    with pytest.raises(RuntimeError, match="reset must be called after"):
        environment.step(0)


@pytest.mark.parametrize("action", [-1, 2])
def test_invalid_action_is_rejected(action: int) -> None:
    environment = LoopingMDP()
    environment.reset()

    with pytest.raises(ValueError, match="action must be 0 or 1"):
        environment.step(action)


def test_time_limit_can_truncate_a_looping_episode() -> None:
    environment = gym.wrappers.TimeLimit(LoopingMDP(), max_episode_steps=2)
    environment.reset()
    environment.step(0)

    state, reward, terminated, truncated, info = environment.step(0)

    assert state == LoopingMDP.DECISION
    assert reward == -0.1
    assert not terminated
    assert truncated
    assert info == {}
