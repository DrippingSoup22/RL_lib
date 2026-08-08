"""Shared execution and aggregation utilities for bandit experiments."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class BanditAgent(Protocol):
    """Small interface needed to measure a bandit agent."""

    estimates: np.ndarray
    counts: np.ndarray

    def select_action(self) -> int: ...

    def update(self, action: int, reward: float) -> None: ...


class BanditEnvironment(Protocol):
    """Environment information needed by a bandit experiment."""

    action_values: np.ndarray

    @property
    def optimal_action(self) -> int: ...

    def step(self, action: int) -> float: ...


@dataclass(frozen=True)
class BanditMeasurements:
    """Per-run measurements with arrays shaped as ``(runs, steps)``."""

    estimation_mse: np.ndarray
    fraction_actions_tried: np.ndarray
    selected_optimal_action: np.ndarray
    instantaneous_regret: np.ndarray
    rewards: np.ndarray


@dataclass(frozen=True)
class BanditConditionResult:
    """Measurements and the parameters that identify one comparison condition."""

    parameters: Mapping[str, float | str]
    measurements: BanditMeasurements


def evaluate_bandit(
    *,
    environment_factory: Callable[[int], BanditEnvironment],
    agent_factory: Callable[[int], BanditAgent],
    runs: int,
    steps: int,
    seed: int,
) -> BanditMeasurements:
    """Evaluate one agent condition using reproducible environment-agent seed pairs.

    The true values are captured before each interaction. This associates the action,
    sampled reward, and post-update estimate with the values that generated that
    interaction, even if a future nonstationary environment drifts inside ``step``.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    shape = (runs, steps)
    estimation_mse = np.empty(shape, dtype=float)
    fraction_actions_tried = np.empty(shape, dtype=float)
    selected_optimal_action = np.empty(shape, dtype=bool)
    instantaneous_regret = np.empty(shape, dtype=float)
    rewards = np.empty(shape, dtype=float)

    for run in range(runs):
        environment = environment_factory(seed + run)
        agent = agent_factory(seed + runs + run)

        for step_index in range(steps):
            action_values = environment.action_values.copy()
            optimal_action = environment.optimal_action
            action = agent.select_action()
            reward = environment.step(action)
            agent.update(action, reward)

            estimation_mse[run, step_index] = np.mean(
                (agent.estimates - action_values) ** 2
            )
            fraction_actions_tried[run, step_index] = np.count_nonzero(
                agent.counts
            ) / len(agent.counts)
            selected_optimal_action[run, step_index] = action == optimal_action
            instantaneous_regret[run, step_index] = (
                action_values[optimal_action] - action_values[action]
            )
            rewards[run, step_index] = reward

    return BanditMeasurements(
        estimation_mse=estimation_mse,
        fraction_actions_tried=fraction_actions_tried,
        selected_optimal_action=selected_optimal_action,
        instantaneous_regret=instantaneous_regret,
        rewards=rewards,
    )


def mean_and_confidence_interval(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean and an approximate 95% confidence interval over runs."""
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("samples must have shape (runs, steps) with at least one run")

    mean = np.mean(samples, axis=0)
    if samples.shape[0] == 1:
        return mean, mean.copy(), mean.copy()

    standard_error = np.std(samples, axis=0, ddof=1) / np.sqrt(samples.shape[0])
    margin = 1.96 * standard_error
    return mean, mean - margin, mean + margin


def rolling_mean(samples: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing mean for each run, using shorter initial windows."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape (runs, steps)")
    if window < 1:
        raise ValueError("window must be at least 1")

    cumulative = np.cumsum(samples, axis=1, dtype=float)
    smoothed = cumulative.copy()
    smoothed[:, window:] -= cumulative[:, :-window]
    counts = np.minimum(np.arange(1, samples.shape[1] + 1), window)
    return smoothed / counts


def _condition_fields(results: Sequence[BanditConditionResult]) -> tuple[str, ...]:
    if not results:
        raise ValueError("at least one condition result is required")
    fields = tuple(results[0].parameters)
    if any(tuple(result.parameters) != fields for result in results[1:]):
        raise ValueError("all condition results must use the same parameter fields")
    return fields


def _aggregate_row(
    measurements: BanditMeasurements,
    step_index: int,
) -> dict[str, float]:
    metric_samples = {
        "estimation_mse": measurements.estimation_mse[:, step_index],
        "fraction_actions_tried": measurements.fraction_actions_tried[:, step_index],
        "optimal_action_rate": measurements.selected_optimal_action[:, step_index],
        "regret": measurements.instantaneous_regret[:, step_index],
        "reward": measurements.rewards[:, step_index],
    }
    row: dict[str, float] = {}
    for name, samples in metric_samples.items():
        row[f"mean_{name}"] = float(np.mean(samples))
        row[f"std_{name}"] = float(np.std(samples))
    return row


def write_convergence_csv(
    path: Path,
    algorithm: str,
    results: Sequence[BanditConditionResult],
) -> None:
    """Write aggregate measurements for every condition and interaction step."""
    condition_fields = _condition_fields(results)
    metric_fields = (
        "mean_estimation_mse",
        "std_estimation_mse",
        "mean_fraction_actions_tried",
        "std_fraction_actions_tried",
        "mean_optimal_action_rate",
        "std_optimal_action_rate",
        "mean_regret",
        "std_regret",
        "mean_reward",
        "std_reward",
    )
    fields = ("algorithm", *condition_fields, "step", *metric_fields)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            steps = result.measurements.rewards.shape[1]
            for step_index in range(steps):
                writer.writerow(
                    {
                        "algorithm": algorithm,
                        **result.parameters,
                        "step": step_index + 1,
                        **_aggregate_row(result.measurements, step_index),
                    }
                )


def write_summary_csv(
    path: Path,
    algorithm: str,
    results: Sequence[BanditConditionResult],
) -> None:
    """Write final-step aggregate measurements for every condition."""
    condition_fields = _condition_fields(results)
    metric_fields = (
        "final_mean_estimation_mse",
        "final_std_estimation_mse",
        "final_mean_fraction_actions_tried",
        "final_std_fraction_actions_tried",
        "final_mean_optimal_action_rate",
        "final_std_optimal_action_rate",
        "final_mean_regret",
        "final_std_regret",
        "final_mean_reward",
        "final_std_reward",
    )
    fields = ("algorithm", *condition_fields, *metric_fields)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            aggregates = _aggregate_row(result.measurements, step_index=-1)
            writer.writerow(
                {
                    "algorithm": algorithm,
                    **result.parameters,
                    **{f"final_{name}": value for name, value in aggregates.items()},
                }
            )
