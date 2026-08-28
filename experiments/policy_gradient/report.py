"""Tuning and standard reports for one policy-gradient algorithm."""

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
from experiments.policy_gradient.configuration import ExperimentConfig
from experiments.summary import SummaryMedia, SummaryTable, write_summary

CSV_FIELDS = (
    "algorithm",
    "seed",
    "checkpoint",
    "evaluation_episode",
    "episode_return",
    "episode_length",
    "success",
    "terminated",
    "truncated",
)


def _seed_means(
    rows: list[dict[str, object]],
    checkpoint: int,
) -> list[dict[str, float | int]]:
    selected = [row for row in rows if int(row["checkpoint"]) == checkpoint]
    result = []
    for seed in sorted({int(row["seed"]) for row in selected}):
        seed_rows = [row for row in selected if int(row["seed"]) == seed]
        result.append(
            {
                "seed": seed,
                "episode_return": float(
                    np.mean([float(row["episode_return"]) for row in seed_rows])
                ),
                "episode_length": float(
                    np.mean([float(row["episode_length"]) for row in seed_rows])
                ),
                "success": float(np.mean([float(row["success"]) for row in seed_rows])),
                "truncated": float(
                    np.mean([float(row["truncated"]) for row in seed_rows])
                ),
            }
        )
    return result


def _aggregates(rows: list[dict[str, object]]) -> list[dict[str, float | int]]:
    result = []
    for checkpoint in EVALUATION_CHECKPOINTS:
        seeds = _seed_means(rows, checkpoint)
        aggregate: dict[str, float | int] = {
            "checkpoint": checkpoint,
            "seeds": len(seeds),
        }
        for metric in ("episode_return", "episode_length", "success", "truncated"):
            values = np.asarray([float(row[metric]) for row in seeds], dtype=float)
            aggregate[metric] = float(np.mean(values))
            aggregate[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        result.append(aggregate)
    return result


def _mean_std(row: dict[str, float | int], metric: str) -> str:
    return f"{float(row[metric]):.3f} +/- {float(row[f'{metric}_std']):.3f}"


def write_report(
    output: Path,
    config: ExperimentConfig,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    """Write the compact tuning or complete single-algorithm standard report."""
    write_csv(output / "metrics.csv", CSV_FIELDS, rows)
    write_metadata(output / "metadata.json", metadata)
    aggregates = _aggregates(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    seed_values = sorted({int(row["seed"]) for row in rows})
    for seed_index, seed in enumerate(seed_values if len(seed_values) > 1 else ()):
        for axis, metric in zip(axes, ("episode_return", "success"), strict=True):
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
        ("episode_return", "success"),
        ("Episode return", "Success rate"),
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
    axes[1].set_ylim(-0.02, 1.02)
    figure.suptitle(f"{config.algorithm} frozen evaluation")
    figure.tight_layout()
    figure.savefig(output / "figures" / "checkpoint_evaluation.png", dpi=150)
    plt.close(figure)

    final_aggregate = aggregates[-1]
    final_rows = []
    if len(seed_values) > 1:
        final_rows.append(
            (
                "All seeds",
                _mean_std(final_aggregate, "episode_return"),
                _mean_std(final_aggregate, "success"),
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
                f"{float(row['episode_length']):.2f}",
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
    if config.preset == "standard":
        tables.append(
            SummaryTable(
                "Checkpoint aggregates",
                ("Training", "Return", "Success", "Episode length", "Truncation"),
                tuple(
                    (
                        f"{row['checkpoint']}%",
                        _mean_std(row, "episode_return"),
                        _mean_std(row, "success"),
                        _mean_std(row, "episode_length"),
                        _mean_std(row, "truncated"),
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
        title=f"Policy gradient on {config.environment}",
        metadata={
            "Environment": config.environment,
            "Mode": config.preset,
            "Algorithm": config.algorithm,
            "Training": f"{config.training_episodes} episodes per seed",
            "Evaluation": f"{config.evaluation_episodes} episodes per checkpoint",
            "Seed trials": list(config.seed_values),
            "Network": list(config.hidden_sizes),
            "Observation normalization": config.observation_normalization,
            "Learning reward scale": config.reward_scale,
            "Evaluation policy": config.evaluation_policy,
            "Optimizer": config.optimizer,
            "Gradient clipping": (
                "off" if config.max_gradient_norm is None else config.max_gradient_norm
            ),
            "Actor learning rate": config.actor_learning_rate,
            "Critic learning rate": config.critic_learning_rate,
            **(
                {
                    "Continuous std / initial": (
                        f"{config.continuous_std} / {config.initial_std:g}"
                    )
                }
                if metadata["action_space_type"] == "bounded continuous"
                else {}
            ),
            **(
                {
                    "PPO batch episodes": config.ppo_batch_episodes,
                    "PPO epochs / minibatch": (
                        f"{config.ppo_update_epochs} / "
                        f"{config.ppo_minibatch_size} transitions"
                    ),
                    "PPO clip / GAE lambda": (
                        f"{config.ppo_clip_ratio:g} / {config.gae_lambda:g}"
                    ),
                }
                if config.algorithm == "ppo"
                else {}
            ),
            "Best model": (
                "best_model.pt" if config.preset == "standard" else "not saved"
            ),
        },
        tables=tables,
        figures=(
            SummaryMedia(
                "Evaluation",
                Path("figures/checkpoint_evaluation.png"),
            ),
        ),
        recordings=recordings,
        compact=config.preset == "tuning",
    )
