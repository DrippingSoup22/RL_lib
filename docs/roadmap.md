# Roadmap

| Order | Family | Primary / challenge environment | Status |
| ---: | --- | --- | --- |
| 1 | Stationary/nonstationary epsilon-greedy and UCB bandits | Stationary / drifting Gaussian bandit | Implemented |
| 2 | First/every visit Monte Carlo prediction and control | `Blackjack-v1` / `Taxi-v4` | Implemented |
| 3 | One-step and bounded-rollout TD prediction, SARSA, Q-learning | `CliffWalking-v1` / configurable `FrozenLake-v1` | Implemented |
| 4 | Semi-gradient one-step and bounded-rollout TD, SARSA, and Q-learning | `Acrobot-v1` / `MountainCar-v0` | Implemented; Acrobot standard validation complete |
| 5 | REINFORCE with baseline, A2C, and A3C | `CartPole-v1` / `Acrobot-v1` | Implemented; A3C standard validation complete |
| 6 | PPO and TRPO | `CartPole-v1` / `Pendulum-v1` | PPO implemented and batched ([batched-ppo.md](batched-ppo.md)); discrete and continuous standard validation complete; TRPO planned |

For each family: implement the variants, add minimal equation-level tests, run
one multi-seed Gymnasium experiment, and generate a deterministic visual
summary. PPO precedes TRPO because it establishes the shared actor-critic
infrastructure more simply.

Acrobot provides the end-to-end validation for function approximation.
MountainCar remains a challenge for the current MLP and epsilon-greedy
exploration setup rather than blocking completion of the family.
The untuned A3C CartPole standard run solved the environment for one of three
seeds. The other two seeds did not solve it, so the run validates the
implementation and experiment workflow while showing that the default settings
are seed-sensitive rather than robustly tuned.
The discrete PPO CartPole standard run solved the environment for all three
paired seeds. Two seeds reached 100% success by the 25% checkpoint and the third
reached it by 50%, using the configuration carried forward from the earlier
one-seed tuning run. The batched PPO repeated this result on the same paired
seeds. With the same settings, the continuous PPO Pendulum standard run learned
on all three seeds, reaching a mean return of -184.3 +/- 17.9 across seeds
from about -1,200 untrained; Pendulum never terminates, so it is judged by
return rather than success.
Standard experiments record frozen evaluation behavior at 25%, 50%, 75%, and
100% and offer only difficulty controls that illuminate the comparison.

For temporal-difference targets, true termination removes the bootstrap term.
Time-limit truncation stops interaction but retains the next-state bootstrap;
it may still count as an unsuccessful evaluation episode.
