import gymnasium as gym
import pytest

from experiments.common import (
    create_run_directory,
    evaluation_checkpoints,
    finite_state_encoding,
    resolve_seed_values,
)


def test_finite_state_encoding_handles_nonzero_discrete_starts() -> None:
    states, encode = finite_state_encoding(gym.spaces.Discrete(3, start=4))

    assert states == 3
    assert [encode(value) for value in (4, 5, 6)] == [0, 1, 2]


def test_finite_state_encoding_flattens_discrete_tuples() -> None:
    states, encode = finite_state_encoding(
        gym.spaces.Tuple(
            (gym.spaces.Discrete(32), gym.spaces.Discrete(11), gym.spaces.Discrete(2))
        )
    )

    assert states == 704
    assert encode((20, 7, 1)) == (20 * 11 + 7) * 2 + 1


def test_finite_state_encoding_rejects_continuous_spaces() -> None:
    with pytest.raises(ValueError, match="tabular"):
        finite_state_encoding(gym.spaces.Box(-1.0, 1.0, shape=(2,)))


def test_development_presets_use_repeatable_paired_seeds() -> None:
    assert resolve_seed_values(3, "tuning", None) == (0, 1, 2)
    assert resolve_seed_values(2, "quick", None) == (0, 1)


def test_explicit_seed_base_reproduces_any_preset() -> None:
    assert resolve_seed_values(3, "standard", 120) == (120, 121, 122)


def test_standard_preset_resolves_one_random_seed_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("experiments.common.secrets.randbelow", lambda _limit: 37)

    assert resolve_seed_values(3, "standard", None) == (37, 38, 39)


@pytest.mark.parametrize(("count", "base"), ((0, 0), (1, -1), (2, 3_999_999)))
def test_invalid_seed_ranges_are_rejected(count: int, base: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        resolve_seed_values(count, "standard", base)


def test_persisted_run_directory_uses_environment_family_and_algorithm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)

    output = create_run_directory(
        "policy_gradient",
        "personal.envs:Author/Control-v0",
        "reinforce_with_baseline",
        "standard",
    )

    assert output.resolve().parent == (
        tmp_path
        / "runs"
        / "personal.envs__Author__Control-v0"
        / "policy_gradient"
        / "reinforce_with_baseline"
    )
    assert output.is_dir()
    assert tuple(output.parent.parent.iterdir()) == (output.parent,)


def test_bandit_run_directory_keeps_the_synthetic_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)

    output = create_run_directory("bandits")

    assert output.resolve().parent == tmp_path / "runs" / "bandits"
    assert output.is_dir()


def test_persisted_run_directory_reuses_parents_for_another_algorithm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)

    ppo_output = create_run_directory(
        "policy_gradient",
        "CartPole-v1",
        "ppo",
        "tuning",
    )
    a3c_output = create_run_directory(
        "policy_gradient",
        "CartPole-v1",
        "a3c",
        "standard",
    )

    family_directory = tmp_path / "runs" / "CartPole-v1" / "policy_gradient"
    assert ppo_output.resolve().parent == family_directory / "ppo"
    assert a3c_output.resolve().parent == family_directory / "a3c"
    assert {path.name for path in family_directory.iterdir()} == {"a3c", "ppo"}


def test_quick_runs_cannot_create_a_persisted_directory() -> None:
    with pytest.raises(ValueError, match="only tuning and standard"):
        create_run_directory("policy_gradient", "CartPole-v1", "ppo", "quick")


def test_quick_mode_evaluates_only_before_and_after_training() -> None:
    assert evaluation_checkpoints("quick") == (0, 100)
    assert evaluation_checkpoints("tuning") == (0, 25, 50, 75, 100)
