"""Tuning and standard reports for one function-approximation algorithm."""

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
    "algorithm",
    "seed",
    "checkpoint",
    "evaluation_kind",
    "policy_checkpoint",
    "evaluation_episode",
    "episode_return",
    "episode_length",
    "success",
    "terminated",
    "truncated",
    "progress",
)


def _seed_means(
    rows: list[dict[str, object]],
    checkpoint: int,
) -> list[dict[str, float | int]]:
    selected = [
        row
        for row in rows
        if row["evaluation_kind"] == "held_out" and int(row["checkpoint"]) == checkpoint
    ]
    result = []
    for seed in sorted({int(row["seed"]) for row in selected}):
        seed_rows = [row for row in selected if int(row["seed"]) == seed]
        values: dict[str, float | int] = {
            "seed": seed,
            "policy_checkpoint": max(
                int(row["policy_checkpoint"]) for row in seed_rows
            ),
        }
        for metric in (
            "episode_return",
            "episode_length",
            "success",
            "truncated",
            "progress",
        ):
            values[metric] = float(np.mean([float(row[metric]) for row in seed_rows]))
        result.append(values)
    return result


def _aggregates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for checkpoint in EVALUATION_CHECKPOINTS:
        seeds = _seed_means(rows, checkpoint)
        aggregate: dict[str, object] = {
            "checkpoint": checkpoint,
            "seeds": len(seeds),
            "policy_checkpoints": sorted(
                {int(row["policy_checkpoint"]) for row in seeds}
            ),
        }
        for metric in (
            "episode_return",
            "episode_length",
            "success",
            "truncated",
            "progress",
        ):
            values = np.asarray([float(row[metric]) for row in seeds], dtype=float)
            aggregate[metric] = float(np.mean(values))
            aggregate[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        result.append(aggregate)
    return result


def _mean_std(row: dict[str, object], metric: str) -> str:
    return f"{float(row[metric]):.3f} +/- {float(row[f'{metric}_std']):.3f}"


def write_report(
    output: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    write_csv(output / "metrics.csv", CSV_FIELDS, rows)
    write_metadata(output / "metadata.json", metadata)
    aggregate = _aggregates(rows)
    progress_label = str(metadata["progress_metric"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    seed_values = sorted({int(row["seed"]) for row in rows})
    for seed_index, seed in enumerate(seed_values if len(seed_values) > 1 else ()):
        for axis, metric in zip(axes, ("success", "progress"), strict=True):
            axis.plot(
                EVALUATION_CHECKPOINTS,
                [
                    float(
                        next(
                            row
                            for row in _seed_means(rows, checkpoint)
                            if int(row["seed"]) == seed
                        )[metric]
                    )
                    for checkpoint in EVALUATION_CHECKPOINTS
                ],
                linestyle=("-", "--", ":", "-.")[seed_index % 4],
                alpha=0.4,
                marker="o",
            )
    for axis, metric, label in zip(
        axes,
        ("success", "progress"),
        ("Success rate", progress_label),
        strict=True,
    ):
        axis.plot(
            EVALUATION_CHECKPOINTS,
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
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].axhline(
        float(metadata["progress_goal"]),
        color="black",
        linestyle="--",
        linewidth=1,
    )
    figure.suptitle(f"{metadata['algorithm']} held-out evaluation")
    figure.tight_layout()
    figure.savefig(output / "figures" / "checkpoint_evaluation.png", dpi=150)
    plt.close(figure)

    final_aggregate = aggregate[-1]
    final_rows = []
    if len(seed_values) > 1:
        final_rows.append(
            (
                "All seeds",
                _mean_std(final_aggregate, "episode_return"),
                _mean_std(final_aggregate, "success"),
                _mean_std(final_aggregate, "progress"),
                _mean_std(final_aggregate, "episode_length"),
                "-",
            )
        )
    selected_seed = metadata["selected_seed"]
    for row in _seed_means(rows, 100):
        final_rows.append(
            (
                f"Seed {row['seed']}",
                f"{float(row['episode_return']):.3f}",
                f"{float(row['success']):.3f}",
                f"{float(row['progress']):.3f}",
                f"{float(row['episode_length']):.2f}",
                "yes" if int(row["seed"]) == selected_seed else "no",
            )
        )
    tables = [
        SummaryTable(
            "Final held-out evaluation",
            (
                "Result",
                "Mean return",
                "Success rate",
                progress_label,
                "Episode length",
                "Recorded",
            ),
            tuple(final_rows),
        )
    ]
    if metadata["preset"] == "standard":
        tables.append(
            SummaryTable(
                "Checkpoint aggregates",
                (
                    "Training",
                    "Selected checkpoint",
                    "Return",
                    "Success",
                    progress_label,
                ),
                tuple(
                    (
                        f"{row['checkpoint']}%",
                        ", ".join(f"{value}%" for value in row["policy_checkpoints"]),
                        _mean_std(row, "episode_return"),
                        _mean_std(row, "success"),
                        _mean_std(row, "progress"),
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
        title=f"Function approximation on {metadata['environment']}",
        metadata={
            "Environment": metadata["environment"],
            "Mode": metadata["preset"],
            "Algorithm": metadata["algorithm"],
            "Training": f"{metadata['training_episodes']} episodes per seed",
            "Evaluation": f"{metadata['evaluation_episodes']} episodes per checkpoint",
            "Seed trials": metadata["seed_values"],
            "Network": metadata["hidden_sizes"],
            "Optimizer": metadata["optimizer"],
            "Learning rate": metadata["learning_rate"],
            "Best model": (
                "best_model.pt" if metadata["preset"] == "standard" else "not saved"
            ),
        },
        tables=tables,
        figures=(
            SummaryMedia(
                "Held-out evaluation",
                Path("figures/checkpoint_evaluation.png"),
            ),
        ),
        recordings=recordings,
        compact=metadata["preset"] == "tuning",
    )
