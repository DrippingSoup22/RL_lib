"""Tuning and standard reports for one temporal-difference algorithm."""

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

CSV_FIELDS = (
    "phase",
    "algorithm",
    "seed",
    "checkpoint",
    "state",
    "V",
    "visits",
    "mean_return",
    "success_rate",
    "mean_episode_length",
    "truncation_rate",
    "mean_cliff_falls",
)
CONTROL_METRICS = (
    "mean_return",
    "success_rate",
    "mean_episode_length",
    "truncation_rate",
    "mean_cliff_falls",
)


def _control_aggregates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for checkpoint in EVALUATION_CHECKPOINTS:
        selected = [row for row in rows if int(row["checkpoint"]) == checkpoint]
        aggregate: dict[str, object] = {"checkpoint": checkpoint}
        for metric in CONTROL_METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            aggregate[metric] = float(np.mean(values))
            aggregate[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        result.append(aggregate)
    return result


def _mean_std(row: dict[str, object], metric: str) -> str:
    return f"{float(row[metric]):.3f} +/- {float(row[f'{metric}_std']):.3f}"


def write_control_report(
    output: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    write_csv(output / "metrics.csv", CSV_FIELDS, rows)
    write_metadata(output / "metadata.json", metadata)
    aggregates = _control_aggregates(rows)
    environment = str(metadata["environment"])
    second_metric = (
        "mean_episode_length" if environment == "CliffWalking-v1" else "success_rate"
    )
    second_label = (
        "Mean episode length" if environment == "CliffWalking-v1" else "Success rate"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    seed_values = sorted({int(row["seed"]) for row in rows})
    for seed_index, seed in enumerate(seed_values if len(seed_values) > 1 else ()):
        selected = [row for row in rows if int(row["seed"]) == seed]
        for axis, metric in zip(
            axes,
            ("mean_return", second_metric),
            strict=True,
        ):
            axis.plot(
                [int(row["checkpoint"]) for row in selected],
                [float(row[metric]) for row in selected],
                linestyle=("-", "--", ":", "-.")[seed_index % 4],
                alpha=0.4,
                marker="o",
            )
    for axis, metric, label in zip(
        axes,
        ("mean_return", second_metric),
        ("Mean return", second_label),
        strict=True,
    ):
        axis.plot(
            EVALUATION_CHECKPOINTS,
            [float(row[metric]) for row in aggregates],
            linewidth=2.6,
            marker="o",
            label="Across-seed mean",
        )
        axis.set_xlabel("Training completed (%)")
        axis.set_ylabel(label)
        axis.set_xticks(EVALUATION_CHECKPOINTS)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"{metadata['algorithm']} evaluation")
    figure.tight_layout()
    figure.savefig(output / "figures" / "control_learning.png", dpi=150)
    plt.close(figure)

    final_aggregate = aggregates[-1]
    final_rows = []
    if len(seed_values) > 1:
        final_rows.append(
            (
                "All seeds",
                _mean_std(final_aggregate, "mean_return"),
                _mean_std(final_aggregate, "success_rate"),
                _mean_std(final_aggregate, "mean_episode_length"),
                "-",
            )
        )
    selected_seed = metadata["selected_seed"]
    for row in rows:
        if int(row["checkpoint"]) == 100:
            final_rows.append(
                (
                    f"Seed {row['seed']}",
                    f"{float(row['mean_return']):.3f}",
                    f"{float(row['success_rate']):.3f}",
                    f"{float(row['mean_episode_length']):.2f}",
                    "yes" if int(row["seed"]) == selected_seed else "no",
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
                ("Training", "Return", "Success", "Episode length", "Truncation"),
                tuple(
                    (
                        f"{row['checkpoint']}%",
                        _mean_std(row, "mean_return"),
                        _mean_std(row, "success_rate"),
                        _mean_std(row, "mean_episode_length"),
                        _mean_std(row, "truncation_rate"),
                    )
                    for row in aggregates
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
        title=f"Temporal difference on {environment}",
        metadata={
            "Environment": environment,
            "Mode": metadata["preset"],
            "Algorithm": metadata["algorithm"],
            "Training": f"{metadata['training_episodes']} episodes per seed",
            "Evaluation": f"{metadata['evaluation_episodes']} episodes per checkpoint",
            "Seed trials": metadata["seed_values"],
            "Learning rate": metadata["learning_rate"],
            "Epsilon": metadata["epsilon"],
            "Rollout length": metadata["rollout_steps"],
            "Best model": (
                "best_model.npz" if metadata["preset"] == "standard" else "not saved"
            ),
        },
        tables=tables,
        figures=(SummaryMedia("Evaluation", Path("figures/control_learning.png")),),
        recordings=recordings,
        compact=metadata["preset"] == "tuning",
    )


def write_prediction_report(
    output: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    write_csv(output / "metrics.csv", CSV_FIELDS, rows)
    write_metadata(output / "metadata.json", metadata)
    states = sorted({int(row["state"]) for row in rows})
    mean_values = np.asarray(
        [
            np.mean([float(row["V"]) for row in rows if int(row["state"]) == state])
            for state in states
        ]
    )
    figure, axis = plt.subplots(figsize=(9, 3.8))
    columns = 12 if metadata["environment"] == "CliffWalking-v1" else 0
    if columns and len(states) % columns == 0:
        image = axis.imshow(mean_values.reshape(-1, columns), cmap="viridis")
        figure.colorbar(image, ax=axis, label="Mean V(s)")
    else:
        axis.plot(states, mean_values)
        axis.set_xlabel("State")
        axis.set_ylabel("Mean V(s)")
    axis.set_title("TD prediction across seed trials")
    figure.tight_layout()
    figure.savefig(output / "figures" / "prediction_values.png", dpi=150)
    plt.close(figure)
    coverage = []
    for seed in metadata["seed_values"]:
        selected = [row for row in rows if int(row["seed"]) == int(seed)]
        visited = [row for row in selected if int(row["visits"]) > 0]
        values = [float(row["V"]) for row in visited]
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
        title=f"TD prediction on {metadata['environment']}",
        metadata={
            "Environment": metadata["environment"],
            "Mode": metadata["preset"],
            "Algorithm": metadata["algorithm"],
            "Episodes per seed": metadata["prediction_episodes"],
            "Seed trials": metadata["seed_values"],
            "Learning rate": metadata["learning_rate"],
            "Rollout length": metadata["rollout_steps"],
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
        figures=(SummaryMedia("State values", Path("figures/prediction_values.png")),),
        compact=metadata["preset"] == "tuning",
    )
