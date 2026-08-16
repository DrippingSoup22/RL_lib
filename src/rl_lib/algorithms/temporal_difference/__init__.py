"""Tabular temporal-difference prediction and control algorithms."""

from rl_lib.algorithms.temporal_difference.control import SARSA, QLearning
from rl_lib.algorithms.temporal_difference.prediction import TDPrediction

__all__ = [
    "QLearning",
    "SARSA",
    "TDPrediction",
]
