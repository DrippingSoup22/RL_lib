# Bandit experiments and metrics

Bandit experiments use the shared evaluator in `evaluation.py`. It runs each
parameter condition over several independent runs, records one measurement per
interaction, and keeps environment and agent seed pairs consistent between
conditions. Pairing the seeds makes differences between algorithms or
hyperparameters less dependent on accidentally easier random bandits.

No single metric gives a complete account of a bandit agent. Reward measures
the payoff actually received, regret measures the quality of the selected
action, optimal-action rate measures exact identification of the best arm, and
MSE measures the internal value estimates. Coverage is retained as a diagnostic
for exploration.

## Measurement timing

At interaction `t`, the evaluator follows this order:

1. Copy the current true action values, `q_t(a)`, and current optimal action.
2. Ask the agent to select `A_t`.
3. Sample `R_t` from the environment.
4. Update the agent's estimate using `A_t` and `R_t`.
5. Record the measurements.

The nonstationary environment samples the reward from `q_t(A_t)` and then
applies its random-walk drift. Consequently, regret, optimal-action selection,
and post-update estimate error are all compared with the values that generated
the current reward. The drifted values become `q_{t+1}` for the next
interaction.

## Recorded metrics

### Reward

The reward is the observed sample `R_t`. Across runs, `mean_reward` estimates
the expected payoff produced by the complete agent and environment interaction.

Reward is the most direct task outcome, but it includes reward noise. In the
nonstationary random-walk environment it also changes scale over time because
the action values spread out and the expected maximum increases. It is
therefore useful for stationary comparisons but less diagnostic of tracking
quality than pseudo-regret in the nonstationary experiment.

Higher reward is better.

### Instantaneous pseudo-regret

For the selected action `A_t`, instantaneous pseudo-regret is

```text
max_a q_t(a) - q_t(A_t)
```

It measures how much expected reward was lost by the action choice, without the
noise of the sampled reward. Zero means the selected arm had an optimal true
value. Unlike reward, regret remains relative to the best arm when the overall
value scale changes.

Pseudo-regret uses the environment's hidden true action values, so it is an
experiment metric rather than something a real agent could observe while
learning. Lower regret is better.

### Optimal-action rate

For each interaction, the evaluator records whether the selected action equals
the environment's current `optimal_action`. Averaging this indicator across
runs gives `mean_optimal_action_rate`, the fraction of runs that selected the
best arm at that step.

This metric is easy to interpret, but it is strict: selecting a nearly optimal
arm counts the same as selecting a very poor arm. Pseudo-regret provides the
missing information about the size of the mistake.

The environment currently exposes one `argmax` action. Exact ties are
negligible for continuously distributed Gaussian values after drift begins,
but the all-zero initialization of the nonstationary bandit is a tie. At that
initial point, pseudo-regret correctly reports zero for every arm while the
optimal-action indicator recognizes only the environment's chosen `argmax`.

Higher optimal-action rate is better.

### Estimation MSE

After the agent update, estimation mean squared error is

```text
(1 / k) * sum_a (Q_{t+1}(a) - q_t(a))^2
```

It measures how accurately the agent tracks the complete vector of true action
values, including arms that were not selected recently. In a nonstationary
problem this reveals whether old observations are being forgotten quickly
enough.

MSE is an internal estimation diagnostic, not a direct policy score. An agent
can have low average MSE but still miss the best arm, or have higher MSE while
preserving the correct ranking and receiving good rewards. The nonstationary
environment starts both `q(a)` and `Q(a)` at zero, so its initial error is
artificially perfect; stochastic updates then raise MSE toward the method's
noise-versus-tracking equilibrium.

Lower MSE is better.

### Fraction of actions tried

The coverage metric is

```text
number of actions with N_t(a) > 0 / k
```

It shows how quickly an agent visits the action space and is particularly
useful when debugging initialization or forced exploration. Once every arm has
been selected it remains equal to one, so it is normally kept in the CSV rather
than occupying a convergence plot.

Higher coverage means broader exploration, but does not by itself imply better
reward or policy quality.

## Which experiments use each metric?

All shared metrics are written for every bandit condition. The figures select
the subset most appropriate to their question:

| Experiment | Figure metrics | Reason |
| --- | --- | --- |
| `epsilon_stationary.py` | Reward, optimal-action rate, MSE | Compares payoff, action identification, and estimate convergence under epsilon and initialization choices. |
| `ucb_stationary.py` | Reward, optimal-action rate, MSE | Compares UCB exploration bonuses after forced initial visits. |
| `epsilon_nonstationary.py` | Pseudo-regret, optimal-action rate, MSE | Compares action quality and value tracking without letting changing reward scale obscure the result. |
| `ucb_nonstationary.py` | Pseudo-regret, optimal-action rate, MSE | Measures standard cumulative-count UCB under drift and separates estimate adaptation from confidence-count adaptation. |

`fraction_actions_tried` and any metric not shown in a particular figure remain
available in the generated CSV files for diagnosis and later analysis.

## Aggregation, uncertainty, and smoothing

`convergence.csv` contains one aggregate row per condition and interaction
step. For every metric it reports the mean and population standard deviation
across independent runs. `summary.csv` contains the same aggregates for only
the final interaction; it is a quick summary, but a single final step should not
replace inspection of the convergence curve in a noisy experiment.

Figures show the across-run mean and an approximate 95% confidence interval:

```text
mean +/- 1.96 * sample_standard_deviation / sqrt(number_of_runs)
```

The nonstationary grid applies a trailing rolling mean independently to each
run before calculating its plotted mean and confidence interval. This makes the
time trend readable without changing the underlying measurements. The rolling
window defaults to 100 interactions and can be disabled with
`--smoothing-window 1`.

CSV values are always unsmoothed.

## Output columns

The common per-step metric columns are:

```text
mean_estimation_mse, std_estimation_mse
mean_fraction_actions_tried, std_fraction_actions_tried
mean_optimal_action_rate, std_optimal_action_rate
mean_regret, std_regret
mean_reward, std_reward
```

Condition columns such as `epsilon`, `step_size`, or the UCB exploration
constant identify which agent produced each row. `summary.csv` prefixes metric
columns with `final_`.
