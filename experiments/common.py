"""Small shared mechanics for experiment runners."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EVALUATION_CHECKPOINTS = (0, 25, 50, 75, 100)
RECORDING_CHECKPOINTS = (25, 50, 75, 100)


def create_run_directory(family: str, environment: str | None = None) -> Path:
    """Create a timestamped run directory and report it immediately."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("runs") / family
    if environment is not None:
        output /= environment
    output /= run_id
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}", flush=True)
    return output


def checkpoint_episode_target(total_episodes: int, checkpoint: int) -> int:
    """Return the cumulative training budget for a percentage checkpoint."""
    if checkpoint == 0:
        return 0
    return max(1, round(total_episodes * checkpoint / 100))


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
