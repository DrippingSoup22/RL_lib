import pytest

gym = pytest.importorskip("gymnasium")

from experiments.config import load_environments  # noqa: E402


@pytest.mark.parametrize("config", load_environments(), ids=lambda config: config.name)
def test_configured_environment_can_step(config) -> None:
    env = gym.make(config.gym_id, **config.make_kwargs)
    try:
        observation, info = env.reset(seed=0)
        next_observation, reward, terminated, truncated, next_info = env.step(
            env.action_space.sample()
        )
    finally:
        env.close()

    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert isinstance(info, dict)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(next_info, dict)

