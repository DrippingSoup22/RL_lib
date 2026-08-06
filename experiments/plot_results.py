"""Aggregate a standard episode CSV and create environment-level graphs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_episodes(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"no episode records found in {path}")
    for row in rows:
        row["seed"] = int(row["seed"])
        row["episode"] = int(row["episode"])
        row["return"] = float(row["return"])
        row["length"] = int(row["length"])
    return rows


def mean_curve(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode"]].append(row["return"])
    episodes = sorted(by_episode)
    return episodes, [statistics.fmean(by_episode[index]) for index in episodes]


def write_environment_result(
    environment: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_algorithm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_algorithm[row["algorithm"]].append(row)

    summary_fields = (
        "environment",
        "algorithm",
        "seeds",
        "episodes",
        "mean_return",
        "std_return",
        "mean_length",
    )
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for algorithm, algorithm_rows in sorted(by_algorithm.items()):
            returns = [row["return"] for row in algorithm_rows]
            writer.writerow(
                {
                    "environment": environment,
                    "algorithm": algorithm,
                    "seeds": len({row["seed"] for row in algorithm_rows}),
                    "episodes": len(algorithm_rows),
                    "mean_return": f"{statistics.fmean(returns):.6g}",
                    "std_return": f"{statistics.pstdev(returns):.6g}",
                    "mean_length": f"{statistics.fmean(
                        row['length'] for row in algorithm_rows
                    ):.6g}",
                }
            )

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for algorithm, algorithm_rows in sorted(by_algorithm.items()):
        episodes, returns = mean_curve(algorithm_rows)
        window = max(1, min(20, math.ceil(len(episodes) / 10)))
        smoothed = [
            statistics.fmean(returns[max(0, index - window + 1) : index + 1])
            for index in range(len(returns))
        ]
        axis.plot(episodes, smoothed, label=algorithm)
    axis.set(
        title=environment.replace("_", " ").title(),
        xlabel="Episode",
        ylabel="Mean return",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing episodes.csv")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to results/<algorithm>/<run-id>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_episodes(args.run_dir / "episodes.csv")
    algorithms = {row["algorithm"] for row in rows}
    algorithm = rows[0]["algorithm"] if len(algorithms) == 1 else "comparison"
    output_root = args.output_root or Path("results") / algorithm / args.run_dir.name

    by_environment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_environment[row["environment"]].append(row)
    for environment, environment_rows in sorted(by_environment.items()):
        write_environment_result(
            environment=environment,
            rows=environment_rows,
            output_dir=output_root / environment,
        )
    print(f"Wrote summaries and graphs to {output_root}")


if __name__ == "__main__":
    main()
