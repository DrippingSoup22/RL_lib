import numpy as np
import pytest
import torch

from rl_lib.models import DiscretePolicyNetwork, GaussianPolicyNetwork
from rl_lib.policies import CategoricalPolicy, SquashedGaussianPolicy


def _linear_categorical_policy() -> CategoricalPolicy:
    model = DiscretePolicyNetwork(2, 3, hidden_sizes=())
    layer = model.network[0]
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                ]
            )
        )
        layer.bias.copy_(torch.tensor([0.5, -0.5, 1.0]))
    return CategoricalPolicy(model)


def test_categorical_policy_samples_action_and_matching_log_probability() -> None:
    policy = _linear_categorical_policy()
    observation = torch.tensor([2.0, 1.0])

    torch.manual_seed(17)
    action, log_probability = policy.sample(observation)

    logits = policy.model(observation)
    expected_log_probability = torch.log_softmax(logits, dim=-1)[action]
    assert action.shape == torch.Size([])
    torch.testing.assert_close(log_probability, expected_log_probability)


def test_categorical_policy_evaluates_batched_actions() -> None:
    policy = _linear_categorical_policy()
    observations = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
    actions = torch.tensor([0, 2])

    log_probabilities, entropy = policy.evaluate_actions(observations, actions)

    logits = policy.model(observations)
    log_probabilities_by_action = torch.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    expected_log_probabilities = log_probabilities_by_action.gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)
    expected_entropy = -(probabilities * log_probabilities_by_action).sum(dim=-1)
    torch.testing.assert_close(log_probabilities, expected_log_probabilities)
    torch.testing.assert_close(entropy, expected_entropy)


def test_categorical_policy_reports_entropy_without_actions() -> None:
    policy = _linear_categorical_policy()
    observations = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])

    entropy = policy.entropy(observations)
    _, expected_entropy = policy.evaluate_actions(
        observations,
        torch.tensor([0, 0]),
    )

    torch.testing.assert_close(entropy, expected_entropy)


def test_categorical_policy_selects_largest_logit_deterministically() -> None:
    policy = _linear_categorical_policy()

    action = policy.deterministic_action(torch.tensor([2.0, 1.0]))
    batched_actions = policy.deterministic_action(
        torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
    )

    assert action.item() == 0
    torch.testing.assert_close(batched_actions, torch.tensor([0, 1]))


def test_squashed_gaussian_policy_calculates_action_transform() -> None:
    model = GaussianPolicyNetwork(3, 2, hidden_sizes=())

    policy = SquashedGaussianPolicy(
        model,
        action_low=[-2.0, 0.0],
        action_high=[2.0, 10.0],
    )

    torch.testing.assert_close(policy.scale, torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(policy.bias, torch.tensor([0.0, 5.0]))
    assert policy.scale.dtype == policy.bias.dtype == torch.float32


def test_squashed_gaussian_policy_transforms_deterministic_actions() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=())
    mean_layer = model.mean_network[0]
    with torch.no_grad():
        mean_layer.weight.copy_(torch.eye(2))
        mean_layer.bias.zero_()
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-2.0, 0.0],
        action_high=[2.0, 10.0],
    )
    central_observation = torch.zeros(2)
    offset_observation = torch.atanh(torch.tensor([0.5, -0.5]))

    central_action = policy.deterministic_action(central_observation)
    batched_actions = policy.deterministic_action(
        torch.stack((central_observation, offset_observation))
    )

    torch.testing.assert_close(central_action, torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(
        batched_actions,
        torch.tensor([[0.0, 5.0], [1.0, 2.5]]),
    )


def test_squashed_gaussian_deterministic_action_preserves_dtype_and_gradients() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=()).to(dtype=torch.float64)
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-2.0, 0.0],
        action_high=[2.0, 10.0],
    )

    action = policy.deterministic_action(torch.zeros(2, dtype=torch.float64))
    action.sum().backward()

    assert action.dtype == torch.float64
    assert model.mean_network[0].bias.grad is not None


def test_squashed_gaussian_policy_samples_and_transforms_latent_action() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=(), initial_std=0.5)
    mean_layer = model.mean_network[0]
    with torch.no_grad():
        mean_layer.weight.copy_(torch.eye(2))
        mean_layer.bias.zero_()
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-2.0, 0.0],
        action_high=[2.0, 10.0],
    )
    observation = torch.tensor([0.25, -0.5])

    torch.manual_seed(17)
    expected_latent_action = torch.distributions.Normal(
        observation,
        torch.full((2,), 0.5),
    ).sample()
    torch.manual_seed(17)
    environment_action, latent_action, log_probability = policy.sample(observation)

    torch.testing.assert_close(latent_action, expected_latent_action)
    torch.testing.assert_close(
        environment_action,
        torch.tensor([2.0, 5.0]) * torch.tanh(expected_latent_action)
        + torch.tensor([0.0, 5.0]),
    )
    expected_log_probability = (
        torch.distributions.Normal(observation, torch.full((2,), 0.5)).log_prob(
            expected_latent_action
        )
        - torch.log1p(-torch.tanh(expected_latent_action).square())
        - torch.log(torch.tensor([2.0, 5.0]))
    ).sum()
    torch.testing.assert_close(log_probability, expected_log_probability)


def test_squashed_gaussian_samples_have_expected_shapes_and_bounds() -> None:
    model = GaussianPolicyNetwork(3, 2, hidden_sizes=())
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-2.0, 0.0],
        action_high=[2.0, 10.0],
    )

    single_action, single_latent_action, single_log_probability = policy.sample(
        torch.zeros(3)
    )
    batched_actions, batched_latent_actions, batched_log_probabilities = policy.sample(
        torch.zeros(32, 3)
    )

    assert single_action.shape == single_latent_action.shape == (2,)
    assert batched_actions.shape == batched_latent_actions.shape == (32, 2)
    assert single_log_probability.shape == ()
    assert batched_log_probabilities.shape == (32,)
    assert torch.all(single_action >= torch.tensor([-2.0, 0.0]))
    assert torch.all(single_action <= torch.tensor([2.0, 10.0]))
    assert torch.all(batched_actions >= torch.tensor([-2.0, 0.0]))
    assert torch.all(batched_actions <= torch.tensor([2.0, 10.0]))


def test_squashed_gaussian_policy_sample_is_seeded_and_detached() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=())
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
    )
    observation = torch.zeros(2)

    torch.manual_seed(23)
    first_action, first_latent_action, first_log_probability = policy.sample(
        observation
    )
    torch.manual_seed(23)
    second_action, second_latent_action, second_log_probability = policy.sample(
        observation
    )

    torch.testing.assert_close(first_action, second_action)
    torch.testing.assert_close(first_latent_action, second_latent_action)
    torch.testing.assert_close(first_log_probability, second_log_probability)
    assert not first_action.requires_grad
    assert not first_latent_action.requires_grad
    assert first_log_probability.requires_grad

    first_log_probability.backward()
    assert model.mean_network[0].bias.grad is not None
    assert model.log_std_network[0].bias.grad is not None


def test_squashed_gaussian_policy_sample_preserves_dtype() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=()).to(dtype=torch.float64)
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
    )

    action, latent_action, log_probability = policy.sample(
        torch.zeros(2, dtype=torch.float64)
    )

    assert action.dtype == latent_action.dtype == log_probability.dtype == torch.float64


def test_squashed_gaussian_policy_evaluates_batched_latent_actions() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=(), initial_std=0.5)
    mean_layer = model.mean_network[0]
    with torch.no_grad():
        mean_layer.weight.copy_(torch.eye(2))
        mean_layer.bias.zero_()
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-2.0, 0.0],
        action_high=[2.0, 10.0],
    )
    observations = torch.tensor([[0.25, -0.5], [-0.25, 0.5]])
    latent_actions = torch.tensor([[0.1, -0.3], [0.7, 0.2]])

    torch.manual_seed(31)
    log_probabilities, entropy = policy.evaluate_actions(
        observations,
        latent_actions,
    )

    expected_log_probabilities = (
        torch.distributions.Normal(
            observations,
            torch.full((2, 2), 0.5),
        ).log_prob(latent_actions)
        - torch.log1p(-torch.tanh(latent_actions).square())
        - torch.log(torch.tensor([2.0, 5.0]))
    ).sum(dim=-1)
    torch.testing.assert_close(log_probabilities, expected_log_probabilities)
    assert entropy.shape == (2,)
    assert entropy.requires_grad


def test_squashed_gaussian_policy_entropy_uses_current_policy_sample() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=())
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
    )
    observations = torch.zeros(4, 2)

    torch.manual_seed(41)
    _, entropy_from_evaluation = policy.evaluate_actions(
        observations,
        torch.zeros(4, 2),
    )
    torch.manual_seed(41)
    entropy = policy.entropy(observations)

    torch.testing.assert_close(entropy, entropy_from_evaluation)
    entropy.mean().backward()
    assert model.mean_network[0].bias.grad is not None
    assert model.log_std_network[0].bias.grad is not None


def test_squashed_gaussian_log_probability_is_stable_near_action_bounds() -> None:
    model = GaussianPolicyNetwork(2, 2, hidden_sizes=())
    policy = SquashedGaussianPolicy(
        model,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
    )

    log_probability, _ = policy.evaluate_actions(
        torch.zeros(2),
        torch.tensor([100.0, -100.0]),
    )

    assert torch.isfinite(log_probability)


@pytest.mark.parametrize(
    ("action_low", "action_high", "message"),
    (
        ([-1.0], [1.0, 1.0], "action_low"),
        ([-1.0, -1.0], [1.0], "action_high"),
        ([[-1.0, -1.0]], [[1.0, 1.0]], "action_low"),
        ([-np.inf, -1.0], [1.0, 1.0], "finite"),
        ([-1.0, -1.0], [1.0, np.nan], "finite"),
        ([-1.0, 1.0], [1.0, 1.0], "greater"),
        ([2.0, -1.0], [1.0, 1.0], "greater"),
    ),
)
def test_squashed_gaussian_policy_rejects_invalid_action_bounds(
    action_low: object,
    action_high: object,
    message: str,
) -> None:
    model = GaussianPolicyNetwork(3, 2, hidden_sizes=())

    with pytest.raises(ValueError, match=message):
        SquashedGaussianPolicy(model, action_low, action_high)
