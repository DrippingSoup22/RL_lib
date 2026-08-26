"""Small shared mechanics for experiment runners."""

from __future__ import annotations

import csv
import json
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

EVALUATION_CHECKPOINTS = (0, 25, 50, 75, 100)
QUICK_EVALUATION_CHECKPOINTS = (0, 100)
RECORDING_CHECKPOINTS = (25, 50, 75, 100)
MAXIMUM_SEED_VALUE = 4_000_000


def finite_state_encoding(
    space: gym.Space,
) -> tuple[int, Callable[[object], int]]:
    """Return a dense index for a Discrete space or a tuple of Discrete spaces."""
    if isinstance(space, gym.spaces.Discrete):
        start = int(space.start)

        def encode_discrete(observation: object) -> int:
            if not space.contains(observation):
                raise ValueError("observation is outside the discrete state space")
            return int(observation) - start

        return int(space.n), encode_discrete

    if isinstance(space, gym.spaces.Tuple) and all(
        isinstance(item, gym.spaces.Discrete) for item in space.spaces
    ):
        discrete_spaces = tuple(space.spaces)
        sizes = tuple(int(item.n) for item in discrete_spaces)
        starts = tuple(int(item.start) for item in discrete_spaces)

        def encode_tuple(observation: object) -> int:
            if not space.contains(observation):
                raise ValueError("observation is outside the finite tuple state space")
            values = tuple(observation)  # type: ignore[arg-type]
            coordinates = tuple(
                int(value) - start for value, start in zip(values, starts, strict=True)
            )
            return int(np.ravel_multi_index(coordinates, sizes))

        return int(np.prod(sizes)), encode_tuple

    raise ValueError(
        "tabular experiments require a Discrete observation space or a tuple "
        "containing only Discrete spaces"
    )


def resolve_seed_values(
    count: int,
    preset: str,
    seed_base: int | None,
) -> tuple[int, ...]:
    """Resolve paired trial seeds, randomizing only standard experiments."""
    if count < 1:
        raise ValueError("seed count must be positive")
    if seed_base is None:
        seed_base = secrets.randbelow(1_000_000) if preset == "standard" else 0
    if seed_base < 0 or seed_base + count > MAXIMUM_SEED_VALUE:
        raise ValueError(
            f"seed base must keep all seeds between 0 and {MAXIMUM_SEED_VALUE - 1}"
        )
    return tuple(range(seed_base, seed_base + count))


def environment_directory_name(environment: str) -> str:
    """Return one safe directory component for a Gymnasium environment id."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "__", environment).strip("._-")
    if not name:
        raise ValueError("environment id must contain a directory-safe character")
    return name


def create_run_directory(
    family: str,
    environment: str | None = None,
    mode: str | None = None,
) -> Path:
    """Create a timestamped persisted-run directory and report it immediately."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if family == "bandits" and environment is None and mode is None:
        output = Path("runs") / family
    else:
        if environment is None:
            raise ValueError("persisted Gymnasium runs require an environment")
        if mode not in ("tuning", "standard"):
            raise ValueError("only tuning and standard experiments persist runs")
        output = Path("runs") / family / environment_directory_name(environment)
    output /= run_id
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}", flush=True)
    return output


def checkpoint_episode_target(total_episodes: int, checkpoint: int) -> int:
    """Return the cumulative training budget for a percentage checkpoint."""
    if checkpoint == 0:
        return 0
    return max(1, round(total_episodes * checkpoint / 100))


def evaluation_checkpoints(preset: str) -> tuple[int, ...]:
    """Use only start/end validation for artifact-free compatibility runs."""
    return QUICK_EVALUATION_CHECKPOINTS if preset == "quick" else EVALUATION_CHECKPOINTS


def recording_title(path: Path, selected_seed: int) -> str:
    """Build a readable recording caption including its selected training seed."""
    algorithm, checkpoint = path.stem.split("_checkpoint_", maxsplit=1)
    label = algorithm.replace("_", " ")
    return f"{label} - {int(checkpoint)}%, seed {selected_seed}"


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Rewrite raw metric rows so completed checkpoints survive interruption."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    """Write complete experiment configuration as deterministic JSON."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def annotated_frame(frame: np.ndarray, label: str) -> Image.Image:
    """Add a compact text label below an RGB environment frame."""
    image = Image.fromarray(frame)
    canvas = Image.new("RGB", (image.width, image.height + 32), "white")
    canvas.paste(image, (0, 0))
    ImageDraw.Draw(canvas).text((10, image.height + 8), label, fill="black")
    return canvas
