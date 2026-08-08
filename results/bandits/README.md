# Bandit experiment results

This report summarizes the stationary and nonstationary bandit experiments.
The reported values are averages over the final 100 interactions, rather than
measurements from only the final interaction. This reduces the influence of
reward and action-selection noise at one particular step.

All experiments use 10-armed Gaussian bandits, 200 independent runs, paired
environment and agent seeds, and reward standard deviation `1.0`. Stationary
experiments run for 500 interactions. Nonstationary experiments run for 3,000
interactions with independent action-value random walks having drift standard
deviation `0.01`.

The metrics are interpreted as follows:

- higher mean reward is better;
- higher optimal-action rate is better;
- lower estimation MSE is better;
- lower pseudo-regret is better.

MSE evaluates the complete vector of action-value estimates, whereas reward,
optimal-action rate, and regret evaluate the decisions produced by those
estimates. Consequently, the configuration with the lowest MSE is not always
the one with the best policy performance.

## Stationary epsilon-greedy

Source run: `20260807T160356000520Z`

The experiment compares epsilon values `0`, `0.01`, and `0.1` with initial
estimates `0`, `1`, and `5`.

With neutral initialization, persistent exploration is clearly beneficial:

| Initial Q | Epsilon | Reward | Optimal-action rate | MSE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 1.003 | 0.325 | 0.860 |
| 0 | 0.01 | 1.193 | 0.474 | 0.721 |
| 0 | 0.1 | **1.317** | **0.698** | **0.212** |

Optimistic initialization improves greedy exploration because unselected arms
remain attractive. Across the complete grid, initial `Q=5` with
`epsilon=0.01` obtains the highest late reward, approximately `1.415`. Initial
`Q=1` with `epsilon=0.1` obtains the highest optimal-action rate, approximately
`0.758`, while initial `Q=5` with `epsilon=0.1` obtains the lowest MSE,
approximately `0.163`.

The experiment therefore demonstrates two valid exploration mechanisms:
explicit random exploration through epsilon and temporary exploration induced
by optimistic initial estimates. Persistent `epsilon=0.1` learns the complete
value vector well but continues taking random actions after learning, which can
slightly reduce its reward.

An earlier run, `20260807T145300159534Z`, has the same configuration, seeds,
and results. It is a duplicate and is not treated as a separate result.

## Stationary UCB

Source run: `20260807T145409312383Z`

The experiment compares UCB exploration constants `c=0`, `c=1`, and `c=2`.
Every arm is visited once before the confidence rule is applied.

| c | Reward | Optimal-action rate | MSE |
| ---: | ---: | ---: | ---: |
| 0 | 1.409 | 0.680 | 0.706 |
| 1 | **1.488** | **0.897** | 0.522 |
| 2 | 1.415 | 0.798 | **0.185** |

At this 500-interaction horizon, `c=1` provides the best balance between
exploration and exploitation. It produces the highest reward and selects the
optimal action most frequently. With `c=0`, the agent often commits too early
after the forced initial visits.

The stronger `c=2` bonus continues sampling uncertain arms and therefore
estimates the complete value vector much more accurately. Its extra exploration
has a short-term policy cost, so its reward and optimal-action rate remain below
those of `c=1` at this horizon.

## Nonstationary epsilon-greedy

Source run: `20260807T164353119432Z`

The experiment compares epsilon values `0`, `0.01`, and `0.1` with a
sample-average update and constant step sizes `0.01` and `0.1`.

The best overall configuration is `epsilon=0.1`, `alpha=0.1`:

| Epsilon | Update | Reward | Optimal-action rate | MSE | Pseudo-regret |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.1 | Sample average | 0.581 | 0.439 | 0.149 | 0.273 |
| 0.1 | alpha=0.01 | 0.606 | 0.472 | 0.191 | 0.249 |
| 0.1 | alpha=0.1 | **0.717** | **0.660** | **0.110** | **0.137** |

Sample-average estimates assign decreasing weight to new rewards, so they
eventually become too slow to follow the random walk. Constant step sizes keep
forgetting old observations. Here, `alpha=0.1` reacts quickly enough to track
the changing values, while `alpha=0.01` is generally too slow over 3,000
interactions.

Exploration is also necessary because an arm that was poor earlier can later
become optimal. The particularly poor combination of `epsilon=0` and the
sample-average update ends with approximately `0.532` pseudo-regret and a
`0.215` optimal-action rate over the final window.

The results also show why MSE must not be interpreted as a policy score in
isolation. With `epsilon=0`, `alpha=0.01` has lower MSE than `alpha=0.1`, but
`alpha=0.1` has lower regret and selects the optimal action more often.

The older run `20260807T162618426749Z` compared only `epsilon=0.1`, the
sample-average update, and `alpha=0.1` over 5,000 interactions. The newer
parameter grid supersedes it.

## Nonstationary UCB

Source run: `20260807T170105254109Z`

The experiment compares cumulative-count UCB constants `c=0`, `c=1`, and
`c=2` with the sample-average update and constant step sizes `0.01` and `0.1`.

The two strongest late configurations use `c=2`:

| c | Update | Reward | Optimal-action rate | MSE | Pseudo-regret |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | Sample average | 0.715 | 0.581 | 0.210 | 0.139 |
| 2 | alpha=0.01 | 0.730 | 0.614 | **0.193** | 0.124 |
| 2 | alpha=0.1 | **0.737** | **0.620** | 0.252 | **0.117** |

The stronger exploration bonus keeps revisiting arms as their values change.
With `c=2`, `alpha=0.1` produces the best reward and lowest regret, while
`alpha=0.01` produces more accurate estimates of the complete value vector.

The worst configuration is `c=0` with sample-average estimates. It obtains
approximately `0.661` pseudo-regret and a `0.151` optimal-action rate over the
final window. It combines stale estimates with no continuing confidence bonus.

This experiment evaluates standard UCB under drift; it is not a fully adapted
nonstationary UCB. A constant step size forgets old rewards in the estimates,
but the confidence bonus still uses cumulative lifetime counts. Discounted or
sliding-window counts are natural future extensions if tracking performance is
studied over longer horizons.

## Overall conclusions

The experiments support four main conclusions:

1. Stationary bandits benefit from directed exploration, whether it comes from
   epsilon, optimistic initialization, or a UCB confidence bonus.
2. Moderate exploration can maximize reward before stronger exploration has
   recovered its information-gathering cost.
3. Nonstationary value estimates need persistent forgetting; sample averages
   become increasingly stale as interaction counts grow.
4. Estimate accuracy and policy quality measure different things and should be
   reported together rather than reduced to one convergence number.

Complete generated measurements and metadata remain under `runs/bandits`.
Those directories are intentionally ignored by Git. This report preserves the
interpretation and authoritative run identifiers; selected figures and compact
CSV summaries can be promoted alongside it when a durable result snapshot is
needed.
