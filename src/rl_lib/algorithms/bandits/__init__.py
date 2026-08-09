"""Multi-armed bandit algorithms."""

from rl_lib.algorithms.bandits.epsilon_greedy import EpsilonGreedyBandits
from rl_lib.algorithms.bandits.ucb import UCBGreedyBandits

__all__ = ["EpsilonGreedyBandits", "UCBGreedyBandits"]
