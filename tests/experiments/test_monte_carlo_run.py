import numpy as np

from experiments.monte_carlo.environments import (
    BLACKJACK_ACTIONS,
    BLACKJACK_STATES,
    encode_blackjack_state,
)
from experiments.monte_carlo.run import run_control


def test_control_runs_on_blackjack_and_saves_the_selected_model(tmp_path) -> None:
    model_path = tmp_path / "best_model.npz"
    rows, selected_seed = run_control(
        "first_visit_control",
        "Blackjack-v1",
        BLACKJACK_STATES,
        BLACKJACK_ACTIONS,
        encode_blackjack_state,
        False,
        1,
        1,
        0.1,
        (7,),
        None,
        (),
        (0, 100),
        None,
        [],
        model_path,
    )

    assert rows
    assert selected_seed == 7
    with np.load(model_path) as saved:
        assert int(saved["seed"]) == 7
        assert saved["Q"].shape == (BLACKJACK_STATES, BLACKJACK_ACTIONS)
