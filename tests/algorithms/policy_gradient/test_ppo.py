import numpy as np
import pytest
import torch

from rl_lib.algorithms.policy_gradient.ppo import (
    PPO,
    PPOActionSample,
    PPOUpdateResult,
)
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork


def make_agent(
    *,
    actor_observation_size: int = 2,
    critic_observation_size: int = 2,
    clip_ratio: float = 0.2,
    entropy_coefficient: float = 0.0,
    seed: int | None = None,
) -> PPO:
    actor_model = DiscretePolicyNetwork(
        actor_observation_size,
        2,
        hidden_sizes=(),
    )
    critic_model = StateValueNetwork(
        critic_observation_size,
        hidden_sizes=(),
    )
    actor_layer = actor_model.network[0]
    critic_layer = critic_model.network[0]
    assert isinstance(actor_layer, torch.nn.Linear)
    assert isinstance(critic_layer, torch.nn.Linear)
    with torch.no_grad():
        actor_layer.weight.zero_()
        actor_layer.bias.zero_()
        critic_layer.weight.zero_()
        critic_layer.bias.fill_(0.5)
        if critic_observation_size == 2:
            critic_layer.weight.copy_(torch.tensor([[1.0, 2.0]]))

    return PPO(
        actor_model,
        torch.optim.SGD(actor_model.parameters(), lr=0.1),
        critic_model,
        torch.optim.SGD(critic_model.parameters(), lr=0.1),
        clip_ratio=clip_ratio,
        entropy_coefficient=entropy_coefficient,
        seed=seed,
    )


def test_ppo_samples_an_action_with_its_frozen_probability_and_value() -> None:
    agent = make_agent()
    actor_before = {
        name: parameter.detach().clone()
        for name, parameter in agent.actor_model.state_dict().items()
    }
    critic_before = {
        name: parameter.detach().clone()
        for name, parameter in agent.critic_model.state_dict().items()
    }

    torch.manual_seed(0)
    sample = agent.sample_action(np.asarray([1.0, 2.0], dtype=np.float32))

    assert isinstance(sample, PPOActionSample)
    assert sample.action in (0, 1)
    assert sample.log_probability == pytest.approx(-np.log(2.0))
    assert sample.value == pytest.approx(5.5)
    assert all(parameter.grad is None for parameter in agent.actor_model.parameters())
    assert all(parameter.grad is None for parameter in agent.critic_model.parameters())
    for name, parameter in agent.actor_model.state_dict().items():
        torch.testing.assert_close(parameter, actor_before[name])
    for name, parameter in agent.critic_model.state_dict().items():
        torch.testing.assert_close(parameter, critic_before[name])


def test_ppo_reads_the_critic_value_for_a_rollout_bootstrap() -> None:
    agent = make_agent()

    assert agent.state_value([3.0, 4.0]) == pytest.approx(11.5)


def test_ppo_select_action_does_not_evaluate_the_critic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent()

    def fail_if_called(_observation: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the critic is unnecessary during action-only selection")

    monkeypatch.setattr(agent.critic_model, "forward", fail_if_called)

    assert agent.select_action([0.0, 0.0]) in (0, 1)


def test_ppo_requires_matching_model_observation_sizes() -> None:
    with pytest.raises(ValueError, match="observation sizes"):
        make_agent(actor_observation_size=2, critic_observation_size=3)


@pytest.mark.parametrize("clip_ratio", (0.0, 1.0, -0.1, 1.1, np.nan, np.inf))
def test_ppo_rejects_an_invalid_clip_ratio(clip_ratio: float) -> None:
    with pytest.raises(ValueError, match="Clip ratio"):
        make_agent(clip_ratio=clip_ratio)


@pytest.mark.parametrize("entropy_coefficient", (-0.1, np.nan, np.inf, -np.inf))
def test_ppo_rejects_an_invalid_entropy_coefficient(
    entropy_coefficient: float,
) -> None:
    with pytest.raises(ValueError, match="Entropy coefficient"):
        make_agent(entropy_coefficient=entropy_coefficient)


@pytest.mark.parametrize(
    "observation",
    (
        1.0,
        [1.0],
        [[1.0, 2.0]],
        [[1.0], [2.0]],
        [1.0, np.nan],
        [1.0, np.inf],
    ),
)
def test_ppo_rejects_an_invalid_observation(observation) -> None:
    with pytest.raises(ValueError, match="Observation|finite"):
        make_agent().sample_action(observation)


@pytest.mark.parametrize(
    ("advantage", "ratio", "expected_actor_loss"),
    (
        (1.0, 2.0, -1.2),
        (-1.0, 0.5, 0.8),
        (1.0, 0.5, -0.5),
        (-1.0, 2.0, 2.0),
    ),
)
def test_ppo_clips_only_advantageous_policy_changes(
    advantage: float,
    ratio: float,
    expected_actor_loss: float,
) -> None:
    agent = make_agent(clip_ratio=0.2)
    current_log_probability = -np.log(2.0)
    old_log_probability = current_log_probability - np.log(ratio)

    result = agent.update_minibatch(
        observations=[[0.0, 0.0]],
        actions=[0],
        old_log_probabilities=[old_log_probability],
        advantages=[advantage],
        return_targets=[0.5],
    )

    assert result.actor_loss == pytest.approx(expected_actor_loss)
    assert result.critic_loss == pytest.approx(0.0)


def test_ppo_updates_actor_and_critic_from_fixed_minibatch_targets() -> None:
    agent = make_agent()
    observation = torch.tensor([0.0, 0.0])
    with torch.no_grad():
        probability_before = torch.softmax(agent.actor_model(observation), dim=-1)[
            0
        ].item()

    result = agent.update_minibatch(
        observations=[[0.0, 0.0]],
        actions=[0],
        old_log_probabilities=[-np.log(2.0)],
        advantages=[1.0],
        return_targets=[1.5],
    )

    with torch.no_grad():
        probability_after = torch.softmax(agent.actor_model(observation), dim=-1)[
            0
        ].item()
    assert isinstance(result, PPOUpdateResult)
    assert result.actor_loss == pytest.approx(-1.0)
    assert result.critic_loss == pytest.approx(0.5)
    assert probability_after > probability_before
    assert agent.state_value([0.0, 0.0]) == pytest.approx(0.6)


def test_ppo_subtracts_the_configured_entropy_bonus() -> None:
    entropy_coefficient = 0.1
    agent = make_agent(entropy_coefficient=entropy_coefficient)

    result = agent.update_minibatch(
        observations=[[0.0, 0.0]],
        actions=[0],
        old_log_probabilities=[-np.log(2.0)],
        advantages=[0.0],
        return_targets=[0.5],
    )

    assert result.entropy == pytest.approx(np.log(2.0))
    assert result.actor_loss == pytest.approx(-entropy_coefficient * np.log(2.0))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("observations", [0.0, 0.0], "shape"),
        ("observations", np.empty((0, 2)), "nonempty"),
        ("actions", [0, 1], "one value"),
        ("actions", [0.5], "integers"),
        ("actions", [2], "action space"),
        ("old_log_probabilities", [], "one value"),
        ("advantages", [np.nan], "finite"),
        ("return_targets", [np.inf], "finite"),
    ),
)
def test_ppo_rejects_an_invalid_minibatch(
    field: str,
    value,
    message: str,
) -> None:
    minibatch = {
        "observations": [[0.0, 0.0]],
        "actions": [0],
        "old_log_probabilities": [-np.log(2.0)],
        "advantages": [1.0],
        "return_targets": [0.5],
    }
    minibatch[field] = value

    with pytest.raises(ValueError, match=message):
        make_agent().update_minibatch(**minibatch)


def test_ppo_normalizes_once_and_uses_every_sample_in_every_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent(seed=7)
    minibatches: list[tuple[np.ndarray, np.ndarray]] = []

    def record_minibatch(
        observations,
        actions,
        old_log_probabilities,
        advantages,
        return_targets,
    ) -> PPOUpdateResult:
        del actions, old_log_probabilities, return_targets
        minibatches.append((np.asarray(observations), np.asarray(advantages)))
        return PPOUpdateResult(0.0, 0.0, 0.0)

    monkeypatch.setattr(agent, "update_minibatch", record_minibatch)

    results = agent.update(
        observations=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        actions=[0, 0, 0],
        old_log_probabilities=[0.0, 0.0, 0.0],
        advantages=[1.0, 2.0, 3.0],
        return_targets=[0.0, 0.0, 0.0],
        update_epochs=2,
        minibatch_size=2,
    )

    assert len(results) == 4
    expected_advantages = {
        0: -np.sqrt(1.5),
        1: 0.0,
        2: np.sqrt(1.5),
    }
    for epoch_start in (0, 2):
        epoch_minibatches = minibatches[epoch_start : epoch_start + 2]
        samples = [
            (int(observation[0]), float(advantage))
            for observations, advantages in epoch_minibatches
            for observation, advantage in zip(observations, advantages, strict=True)
        ]
        assert sorted(index for index, _advantage in samples) == [0, 1, 2]
        for index, advantage in samples:
            assert advantage == pytest.approx(expected_advantages[index])


def test_ppo_keeps_constant_advantages_instead_of_removing_the_actor_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = make_agent(seed=0)
    received_advantages: list[np.ndarray] = []

    def record_minibatch(
        observations,
        actions,
        old_log_probabilities,
        advantages,
        return_targets,
    ) -> PPOUpdateResult:
        del observations, actions, old_log_probabilities, return_targets
        received_advantages.append(np.asarray(advantages))
        return PPOUpdateResult(0.0, 0.0, 0.0)

    monkeypatch.setattr(agent, "update_minibatch", record_minibatch)

    agent.update(
        observations=[[0.0, 0.0], [1.0, 0.0]],
        actions=[0, 1],
        old_log_probabilities=[0.0, 0.0],
        advantages=[2.0, 2.0],
        return_targets=[0.0, 0.0],
        update_epochs=1,
        minibatch_size=2,
    )

    np.testing.assert_allclose(received_advantages[0], [2.0, 2.0])


@pytest.mark.parametrize(
    ("update_epochs", "minibatch_size", "message"),
    ((0, 1, "epochs"), (1, 0, "Minibatch")),
)
def test_ppo_rejects_invalid_batch_update_counts(
    update_epochs: int,
    minibatch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_agent().update(
            observations=[[0.0, 0.0]],
            actions=[0],
            old_log_probabilities=[0.0],
            advantages=[1.0],
            return_targets=[0.0],
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        )
