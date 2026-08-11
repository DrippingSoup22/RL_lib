# Roadmap

| Order | Family | Primary / challenge environment | Status |
| ---: | --- | --- | --- |
| 1 | Stationary/nonstationary epsilon-greedy and UCB bandits | Stationary / drifting Gaussian bandit | Implemented |
| 2 | First/every visit Monte Carlo prediction and control | `Blackjack-v1` / `Taxi-v4` | Implemented |
| 3 | TD(0), SARSA, Q-learning | `CliffWalking-v1` / configurable `FrozenLake-v1` | Implemented |
| 4 | Semi-gradient SARSA and Q-learning | `MountainCar-v0` / `Acrobot-v1` | Planned |
| 5 | REINFORCE with baseline and A2C | `CartPole-v1` / `Acrobot-v1` | Planned |
| 6 | A3C, PPO, TRPO | `CartPole-v1` / `Pendulum-v1` | Planned |

For each family: implement the variants, add minimal equation-level tests, run
one multi-seed Gymnasium experiment, and generate a deterministic visual
summary. PPO precedes TRPO because it establishes the shared actor-critic
infrastructure more simply.
Standard experiments record frozen evaluation behavior at 25%, 50%, 75%, and
100% and offer only difficulty controls that illuminate the comparison.

For temporal-difference targets, true termination removes the bootstrap term.
Time-limit truncation stops interaction but retains the next-state bootstrap;
it may still count as an unsuccessful evaluation episode.
