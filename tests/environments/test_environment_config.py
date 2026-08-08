from experiments.config import load_environments
from experiments.runners.watch_environment import resolve_render_mode


def test_environment_names_are_unique() -> None:
    configs = load_environments()

    assert configs
    assert len({config.name for config in configs}) == len(configs)


def test_each_environment_has_reproducible_baseline_settings() -> None:
    for config in load_environments():
        assert config.baseline_seeds
        assert config.baseline_episodes > 0


def test_frozen_lake_variants_isolate_transition_randomness() -> None:
    deterministic = load_environments("frozen_lake")[0]
    stochastic = load_environments("frozen_lake_slippery")[0]

    assert deterministic.gym_id == stochastic.gym_id == "FrozenLake-v1"
    assert deterministic.make_kwargs["is_slippery"] is False
    assert stochastic.make_kwargs["is_slippery"] is True


def test_auto_rendering_uses_terminal_only_for_toy_text() -> None:
    assert resolve_render_mode(load_environments("frozen_lake")[0], "auto") == "ansi"
    assert resolve_render_mode(load_environments("cart_pole")[0], "auto") == "human"
