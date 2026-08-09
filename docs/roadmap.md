# Roadmap

| Order | Family | Environment | Status |
| ---: | --- | --- | --- |
| 1 | Stationary/nonstationary epsilon-greedy and UCB bandits | Gymnasium-compatible Gaussian bandit | Implemented |
| 2 | First/every visit Monte Carlo prediction and control | `Blackjack-v1` | Implemented |
| 3 | TD(0), SARSA, Q-learning | `FrozenLake-v1`, `CliffWalking-v1` | Planned |
| 4 | Semi-gradient SARSA and Q-learning | `MountainCar-v0` | Planned |
| 5 | REINFORCE with baseline and A2C | `CartPole-v1` | Planned |
| 6 | A3C, PPO, TRPO | `CartPole-v1` | Planned |

For each family: implement the variants, add minimal equation-level tests, run
one multi-seed Gymnasium experiment, and record a concise report. PPO precedes
TRPO because it establishes the shared actor-critic infrastructure more simply.
