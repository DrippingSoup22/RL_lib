import math

import pytest
import torch

from rl_lib.networks import CategoricalPolicyNetwork, GaussianPolicyNetwork
from rl_lib.policies import CategoricalPolicy, SquashedGaussianPolicy

ACTION_SCALE = torch.tensor([2.0, 5.0])
ACTION_BIAS = torch.tensor([0.0, 5.0])


def _linear_categorical_policy(generator=None) -> CategoricalPolicy:
    network = CategoricalPolicyNetwork(2, 3, hidden_sizes=())
    layer = network.network[0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
        layer.bias.copy_(torch.tensor([0.5, -0.5, 1.0]))
    return CategoricalPolicy(network, generator=generator)


def _linear_squashed_gaussian_policy(generator=None) -> SquashedGaussianPolicy:
    """Mean equal to the observation, spread 0.5, bounds [-2, 2] and [0, 10]."""
    network = GaussianPolicyNetwork(2, 2, hidden_sizes=(), initial_std=0.5)
    mean_layer = network.mean_network[0]
    with torch.no_grad():
        mean_layer.weight.copy_(torch.eye(2))
        mean_layer.bias.zero_()
    return SquashedGaussianPolicy(
        network, action_low=[-2.0, 0.0], action_high=[2.0, 10.0], generator=generator
    )


def _expected_log_probabilities(observations, latent_actions) -> torch.Tensor:
    """Normal density, corrected for tanh and for scaling to the bounds."""
    return (
        torch.distributions.Normal(observations, 0.5).log_prob(latent_actions)
        - torch.log1p(-torch.tanh(latent_actions).square())
        - torch.log(ACTION_SCALE)
    ).sum(dim=-1)


def test_categorical_log_probabilities_and_entropy_follow_the_logits() -> None:
    policy = _linear_categorical_policy(torch.Generator().manual_seed(17))
    observations = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
    log_probabilities_by_action = torch.log_softmax(policy.network(observations), -1)
    expected_entropy = -(
        log_probabilities_by_action.exp() * log_probabilities_by_action
    ).sum(dim=-1)

    def expected_log_probabilities(actions: torch.Tensor) -> torch.Tensor:
        return log_probabilities_by_action.gather(1, actions[:, None]).squeeze(1)

    actions, log_probabilities = policy.sample(observations)
    torch.testing.assert_close(log_probabilities, expected_log_probabilities(actions))

    stored_actions = torch.tensor([0, 2])
    log_probabilities, entropy = policy.evaluate_actions(observations, stored_actions)
    torch.testing.assert_close(
        log_probabilities, expected_log_probabilities(stored_actions)
    )
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(policy.entropy(observations), expected_entropy)


def test_categorical_policy_selects_the_largest_logit_deterministically() -> None:
    policy = _linear_categorical_policy()

    actions = policy.deterministic_action(torch.tensor([[2.0, 1.0], [-1.0, 3.0]]))

    torch.testing.assert_close(actions, torch.tensor([0, 1]))


def test_squashed_gaussian_deterministic_action_is_the_transformed_mean() -> None:
    policy = _linear_squashed_gaussian_policy()
    torch.testing.assert_close(policy.scale, ACTION_SCALE)
    torch.testing.assert_close(policy.bias, ACTION_BIAS)

    observations = torch.stack((torch.zeros(2), torch.atanh(torch.tensor([0.5, -0.5]))))
    actions = policy.deterministic_action(observations)

    torch.testing.assert_close(actions, torch.tensor([[0.0, 5.0], [1.0, 2.5]]))


def test_squashed_gaussian_sample_transforms_a_latent_normal_sample() -> None:
    policy = _linear_squashed_gaussian_policy(torch.Generator().manual_seed(17))
    observations = torch.tensor([[0.25, -0.5], [-0.25, 0.5]])

    # The mean plus the spread times standard normal noise from the generator.
    noise = torch.randn(2, 2, generator=torch.Generator().manual_seed(17))
    expected_latent_actions = observations + 0.5 * noise
    environment_actions, latent_actions, log_probabilities = policy.sample(observations)

    torch.testing.assert_close(latent_actions, expected_latent_actions)
    torch.testing.assert_close(
        environment_actions,
        ACTION_SCALE * torch.tanh(expected_latent_actions) + ACTION_BIAS,
    )
    torch.testing.assert_close(
        log_probabilities,
        _expected_log_probabilities(observations, expected_latent_actions),
    )
    # Actions are data; only the log-probability carries gradients to the network.
    assert not environment_actions.requires_grad
    assert not latent_actions.requires_grad
    log_probabilities.sum().backward()
    assert policy.network.mean_network[0].bias.grad is not None
    assert policy.network.log_std_network[0].bias.grad is not None


def test_squashed_gaussian_evaluates_stored_latent_actions_and_entropy() -> None:
    generator = torch.Generator()
    policy = _linear_squashed_gaussian_policy(generator)
    observations = torch.tensor([[0.25, -0.5], [-0.25, 0.5]])
    latent_actions = torch.tensor([[0.1, -0.3], [0.7, 0.2]])

    generator.manual_seed(31)
    log_probabilities, entropy = policy.evaluate_actions(observations, latent_actions)
    generator.manual_seed(31)
    entropy_alone = policy.entropy(observations)

    torch.testing.assert_close(
        log_probabilities, _expected_log_probabilities(observations, latent_actions)
    )
    # Entropy comes from a fresh sample of the current policy, not from the
    # stored actions, and carries gradients for the entropy bonus.
    torch.testing.assert_close(entropy, entropy_alone)
    entropy.mean().backward()
    assert policy.network.mean_network[0].bias.grad is not None
    assert policy.network.log_std_network[0].bias.grad is not None


def test_squashed_gaussian_log_probability_is_stable_near_action_bounds() -> None:
    policy = _linear_squashed_gaussian_policy()

    log_probability, _ = policy.evaluate_actions(
        torch.zeros(2), torch.tensor([100.0, -100.0])
    )

    assert torch.isfinite(log_probability)


def test_squashed_gaussian_policy_rejects_invalid_action_bounds() -> None:
    network = GaussianPolicyNetwork(3, 2, hidden_sizes=())
    for action_low, action_high, message in (
        ([-1.0], [1.0, 1.0], "action_low"),
        ([-1.0, -1.0], [1.0], "action_high"),
        ([-math.inf, -1.0], [1.0, 1.0], "finite"),
        ([2.0, -1.0], [1.0, 1.0], "greater"),
    ):
        with pytest.raises(ValueError, match=message):
            SquashedGaussianPolicy(network, action_low, action_high)
