import pytest

from rl_lib.algorithms.bandits import EpsilonGreedyBandits, UCBGreedyBandits


def test_epsilon_greedy_uses_incremental_sample_average() -> None:
    agent = EpsilonGreedyBandits(k=2, epsilon=0.0, seed=0)
    for reward in (2.0, 4.0, 6.0):
        agent.update(1, reward)
    assert agent.estimates[1] == pytest.approx(4.0)
    assert agent.counts[1] == 3


def test_constant_step_size_forgets_old_rewards() -> None:
    agent = EpsilonGreedyBandits(k=2, step_size=0.5)
    agent.update(0, 2.0)
    agent.update(0, 4.0)
    assert agent.estimates[0] == pytest.approx(2.5)


def test_greedy_action_has_the_largest_estimate() -> None:
    agent = EpsilonGreedyBandits(k=3, epsilon=0.0, seed=0)
    agent.estimates[:] = [1.0, 3.0, 2.0]
    assert agent.select_action() == 1


def test_ucb_visits_every_action_before_reusing_one() -> None:
    agent = UCBGreedyBandits(k=4, seed=0)
    actions = []
    for _ in range(4):
        action = agent.select_action()
        actions.append(action)
        agent.update(action, 0.0)
    assert set(actions) == set(range(4))


@pytest.mark.parametrize("agent_class", [EpsilonGreedyBandits, UCBGreedyBandits])
def test_bandits_reject_invalid_actions(agent_class: type) -> None:
    with pytest.raises(ValueError, match="action"):
        agent_class(k=2).update(2, 0.0)
