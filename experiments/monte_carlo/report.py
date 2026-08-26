"""Focused tuning and standard reports for one Monte Carlo variant."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    EVALUATION_CHECKPOINTS,
    recording_title,
    write_csv,
    write_metadata,
)
from experiments.summary import SummaryMedia, SummaryTable, write_summary


def _prediction_figure(
    path: Path,
    rows: list[dict[str, object]],
    environment: str,
) -> None:
    states = sorted({int(row["state"]) for row in rows})
    values = {
        state: float(
            np.mean([float(row["value"]) for row in rows if int(row["state"]) == state])
        )
        for state in states
    }
    if environment == "Blackjack-v1":
        limit = max(0.01, max(abs(value) for value in values.values()))
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        image = None
        for axis, usable_ace in zip(axes, (0, 1), strict=True):
            grid = np.empty((10, 10), dtype=float)
            for state, value in values.items():
                player = state // 22
                dealer = (state % 22) // 2
                ace = state % 2
                if 12 <= player <= 21 and 1 <= dealer <= 10 and ace == usable_ace:
                    grid[player - 12, dealer - 1] = value
            image = axis.imshow(
                grid,
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            axis.set_title("Usable ace" if usable_ace else "No usable ace")
            axis.set_xticks(range(10), range(1, 11))
            axis.set_yticks(range(10), range(12, 22))
            axis.set_xlabel("Dealer showing")
            axis.set_ylabel("Player total")
        figure.colorbar(image, ax=axes, label="Mean state value")
    else:
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(states, [values[state] for state in states])
        axis.set_xlabel("State")
        axis.set_ylabel("Mean value")
        axis.grid(alpha=0.25)
    figure.suptitle("Fixed-policy Monte Carlo prediction")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_prediction_report(
    output: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    write_csv(output / "metrics.csv", tuple(rows[0]), rows)
    write_metadata(output / "metadata.json", metadata)
    figure_path = output / "figures" / "prediction_values.png"
    _prediction_figure(figure_path, rows, str(metadata["environment"]))
    coverage = []
    for seed in metadata["seed_values"]:
        selected = [row for row in rows if int(row["seed"]) == int(seed)]
        visited = [row for row in selected if int(row["visits"]) > 0]
        values = [float(row["value"]) for row in visited]
        coverage.append(
            (
                seed,
                len(visited),
                sum(int(row["visits"]) for row in selected),
                f"{min(values):.3f} to {max(values):.3f}" if values else "not visited",
            )
        )
    write_summary(
        output / "summary.html",
        title=f"Monte Carlo prediction on {metadata['environment']}",
        metadata={
            "Environment": metadata["environment"],
            "Mode": metadata["preset"],
            "Algorithm": metadata["algorithm"],
            "Episodes per seed": metadata["prediction_episodes"],
            "Seed trials": metadata["seed_values"],
            "Best model": (
                "best_model.npz" if metadata["preset"] == "standard" else "not saved"
            ),
        },
        tables=(
            SummaryTable(
                "Prediction coverage",
                ("Seed", "Visited states", "Total visits", "Value range"),
                tuple(coverage),
            ),
        ),
        figures=(
            SummaryMedia(
                "State-value estimates",
                Path("figures/prediction_values.png"),
            ),
        ),
        compact=metadata["preset"] == "tuning",
    )


def _mean_and_std(row: dict[str, object], mean: str, std: str) -> str:
    return f"{float(row[mean]):.3f} +/- {float(row[std]):.3f}"


def write_control_report(
    output: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    write_csv(output / "metrics.csv", tuple(rows[0]), rows)
    write_metadata(output / "metadata.json", metadata)
    environment = str(metadata["environment"])
    algorithm = str(metadata["algorithm"])
    aggregate = [row for row in rows if row["phase"] == "control"]
    seeds = [row for row in rows if row["phase"] == "control_seed"]
    metrics = [("mean_return", "Mean return")]
    if aggregate[0]["success_rate"] != "":
        metrics.append(("success_rate", "Success rate"))
    else:
        metrics.append(("mean_episode_length", "Episode length"))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    seed_values = sorted({int(row["seed"]) for row in seeds})
    for seed_index, seed in enumerate(seed_values if len(seed_values) > 1 else ()):
        selected = [row for row in seeds if int(row["seed"]) == seed]
        for axis, (metric, _) in zip(axes, metrics, strict=True):
            axis.plot(
                [int(row["checkpoint"]) for row in selected],
                [float(row[metric]) for row in selected],
                linestyle=("-", "--", ":", "-.")[seed_index % 4],
                alpha=0.4,
                marker="o",
            )
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        axis.plot(
            [int(row["checkpoint"]) for row in aggregate],
            [float(row[metric]) for row in aggregate],
            linewidth=2.6,
            marker="o",
            label="Across-seed mean",
        )
        axis.set_xlabel("Training completed (%)")
        axis.set_ylabel(label)
        axis.set_xticks(EVALUATION_CHECKPOINTS)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"{algorithm.replace('_', ' ').title()} evaluation")
    figure.tight_layout()
    figure.savefig(output / "figures" / "control_learning.png", dpi=150)
    plt.close(figure)

    final_aggregate = next(row for row in aggregate if int(row["checkpoint"]) == 100)
    final_rows = []
    if len(seed_values) > 1:
        final_rows.append(
            (
                "All seeds",
                _mean_and_std(final_aggregate, "mean_return", "return_std"),
                (
                    _mean_and_std(final_aggregate, "success_rate", "success_rate_std")
                    if final_aggregate["success_rate"] != ""
                    else "not defined"
                ),
                _mean_and_std(
                    final_aggregate,
                    "mean_episode_length",
                    "episode_length_std",
                ),
                "-",
            )
        )
    selected_recording_seed = metadata["selected_seed"]
    for row in seeds:
        if int(row["checkpoint"]) == 100:
            final_rows.append(
                (
                    f"Seed {row['seed']}",
                    f"{float(row['mean_return']):.3f}",
                    (
                        f"{float(row['success_rate']):.3f}"
                        if row["success_rate"] != ""
                        else "not defined"
                    ),
                    f"{float(row['mean_episode_length']):.2f}",
                    "yes" if int(row["seed"]) == selected_recording_seed else "no",
                )
            )
    tables = [
        SummaryTable(
            "Final evaluation",
            ("Result", "Mean return", "Success rate", "Episode length", "Recorded"),
            tuple(final_rows),
        )
    ]
    if metadata["preset"] == "standard":
        tables.append(
            SummaryTable(
                "Checkpoint aggregates",
                ("Training", "Mean return", "Success rate", "Episode length"),
                tuple(
                    (
                        f"{row['checkpoint']}%",
                        _mean_and_std(row, "mean_return", "return_std"),
                        (
                            _mean_and_std(row, "success_rate", "success_rate_std")
                            if row["success_rate"] != ""
                            else "not defined"
                        ),
                        _mean_and_std(
                            row,
                            "mean_episode_length",
                            "episode_length_std",
                        ),
                    )
                    for row in aggregate
                ),
                collapsed=True,
            )
        )
    recordings = tuple(
        SummaryMedia(
            recording_title(path, int(metadata["selected_seed"])),
            path.relative_to(output),
        )
        for path in sorted((output / "recordings").glob("*.gif"))
    )
    write_summary(
        output / "summary.html",
        title=f"Monte Carlo control on {environment}",
        metadata={
            "Environment": environment,
            "Mode": metadata["preset"],
            "Algorithm": algorithm,
            "Training": f"{metadata['training_episodes']} episodes per seed",
            "Evaluation": f"{metadata['evaluation_episodes']} episodes per checkpoint",
            "Seed trials": metadata["seed_values"],
            "Epsilon": metadata["epsilon"],
            "Best model": (
                "best_model.npz" if metadata["preset"] == "standard" else "not saved"
            ),
        },
        tables=tables,
        figures=(SummaryMedia("Evaluation", Path("figures/control_learning.png")),),
        recordings=recordings,
        compact=metadata["preset"] == "tuning",
    )
