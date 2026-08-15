import pytest

from experiments.function_approximation.diagnostics import (
    EpisodeDiagnosticTracker,
)


def test_episode_diagnostics_summarize_learning_signals() -> None:
    tracker = EpisodeDiagnosticTracker(number_of_actions=3)
    tracker.record_step(
        reward=-1.0,
        action=0,
        td_error=-2.0,
        action_values=(-3.0, -1.0, -2.0),
    )
    tracker.record_step(
        reward=2.0,
        action=2,
        td_error=4.0,
        action_values=(1.0, 5.0, -1.0),
    )

    result = tracker.finish()

    assert result.episode_return == 1.0
    assert result.episode_length == 2
    assert result.mean_absolute_td_error == 3.0
    assert result.maximum_absolute_td_error == 4.0
    assert result.mean_absolute_q_value == pytest.approx(13 / 6)
    assert result.maximum_absolute_q_value == 5.0
    assert result.mean_action_gap == 2.5
    assert result.dominant_action_fraction == 0.5
    assert result.action_counts == (1, 0, 1)


def test_episode_diagnostics_validate_dimensions_and_nonempty_results() -> None:
    with pytest.raises(ValueError, match="number of actions"):
        EpisodeDiagnosticTracker(number_of_actions=0)

    tracker = EpisodeDiagnosticTracker(number_of_actions=2)
    with pytest.raises(ValueError, match="recorded step"):
        tracker.finish()
    with pytest.raises(ValueError, match="one value per action"):
        tracker.record_step(
            reward=0.0,
            action=0,
            td_error=0.0,
            action_values=(1.0,),
        )
