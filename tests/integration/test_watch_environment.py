import pytest

pytest.importorskip("gymnasium")

from experiments.config import load_environments  # noqa: E402
from experiments.runners.watch_environment import watch_random_agent  # noqa: E402


def test_watch_random_agent_can_run_without_a_window() -> None:
    config = load_environments("frozen_lake")[0]

    returns = watch_random_agent(config, episodes=2, seed=0, render_mode="rgb_array")

    assert len(returns) == 2
