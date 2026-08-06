from experiments.config import load_environments


def test_environment_names_are_unique() -> None:
    configs = load_environments()

    assert configs
    assert len({config.name for config in configs}) == len(configs)


def test_each_environment_has_reproducible_baseline_settings() -> None:
    for config in load_environments():
        assert config.baseline_seeds
        assert config.baseline_episodes > 0

