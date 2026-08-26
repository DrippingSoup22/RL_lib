import numpy as np
import pytest

from experiments.temporal_difference.run import (
    make_environment,
    run_prediction,
    train_episode,
)
from rl_lib.algorithms.temporal_difference import SARSA, QLearning


@pytest.mark.parametrize(
    ("algorithm", "agent_class"),
    (("sarsa", SARSA), ("q_learning", QLearning)),
)
def test_tabular_runner_updates_a_short_rollout_after_truncation(
    algorithm: str,
    agent_class: type[SARSA] | type[QLearning],
) -> None:
    env = make_environment("CliffWalking-v1", {"max_episode_steps": 1})
    agent = agent_class(48, 4, learning_rate=0.1, seed=0)

    train_episode(env, agent, algorithm, seed=0, rollout_steps=3)

    assert np.any(agent.Q != 0.0)
    env.close()


def test_prediction_rows_use_the_cli_algorithm_name() -> None:
    rows = run_prediction(
        "FrozenLake-v1",
        {
            "desc": ["SF", "FG"],
            "is_slippery": False,
            "max_episode_steps": 4,
        },
        number_of_states=4,
        number_of_actions=4,
        episodes=1,
        learning_rate=0.1,
        discount=1.0,
        rollout_steps=1,
        seed=0,
    )

    assert {row["algorithm"] for row in rows} == {"td_prediction"}
