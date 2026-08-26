# Roadmap

| Order | Family | Primary / challenge environment | Status |
| ---: | --- | --- | --- |
| 1 | Stationary/nonstationary epsilon-greedy and UCB bandits | Stationary / drifting Gaussian bandit | Implemented |
| 2 | First/every visit Monte Carlo prediction and control | `Blackjack-v1` / `Taxi-v4` | Implemented |
| 3 | One-step and bounded-rollout TD prediction, SARSA, Q-learning | `CliffWalking-v1` / configurable `FrozenLake-v1` | Implemented |
| 4 | Semi-gradient one-step and bounded-rollout TD, SARSA, and Q-learning | `Acrobot-v1` / `MountainCar-v0` | Implemented; Acrobot standard validation complete |
| 5 | REINFORCE with baseline, A2C, and A3C | `CartPole-v1` / `Acrobot-v1` | Implemented; A3C standard validation pending |
| 6 | PPO and TRPO | `CartPole-v1` / `Pendulum-v1` | Discrete PPO implemented; continuous PPO and TRPO planned |

For each family: implement the variants, add minimal equation-level tests, run
one multi-seed Gymnasium experiment, and generate a deterministic visual
summary. PPO precedes TRPO because it establishes the shared actor-critic
infrastructure more simply.

Acrobot provides the end-to-end validation for function approximation.
MountainCar remains a challenge for the current MLP and epsilon-greedy
exploration setup rather than blocking completion of the family.
Standard experiments record frozen evaluation behavior at 25%, 50%, 75%, and
100% and offer only difficulty controls that illuminate the comparison.

For temporal-difference targets, true termination removes the bootstrap term.
Time-limit truncation stops interaction but retains the next-state bootstrap;
it may still count as an unsuccessful evaluation episode.
