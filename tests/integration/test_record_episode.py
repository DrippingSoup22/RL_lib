from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("gymnasium")

from experiments.config import load_environments  # noqa: E402
from experiments.runners.record_episode import record_random_episode  # noqa: E402


def test_record_random_episode_creates_animated_gif(tmp_path: Path) -> None:
    config = load_environments("frozen_lake")[0]
    output_path = tmp_path / "episode.gif"

    episode_return = record_random_episode(config, output_path, seed=0)

    with Image.open(output_path) as recording:
        assert recording.format == "GIF"
        assert recording.n_frames > 1
    assert isinstance(episode_return, float)
